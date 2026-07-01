from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import LEGACY_TOKEN_ISSUER, TOKEN_ISSUER, normalize_audiences

TokenKind = Literal["run_access", "llm_gateway", "web_gateway", "task_callback", "capability"]
TOKEN_HEADER_TYPE = "MAK"
LEGACY_TOKEN_HEADER_TYPE = "NAR"
ACCEPTED_TOKEN_HEADER_TYPES = (TOKEN_HEADER_TYPE, LEGACY_TOKEN_HEADER_TYPE)


class TokenError(NativeAgentError):
    pass


@dataclass(frozen=True)
class TokenClaims:
    kind: TokenKind
    audience: str
    run_id: str
    tenant_id: str
    user_id: str
    issued_at: int
    expires_at: int
    token_id: str = field(default_factory=lambda: secrets.token_hex(12))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "typ": self.kind,
            "aud": self.audience,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "iat": self.issued_at,
            "exp": self.expires_at,
            "jti": self.token_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TokenClaims:
        return cls(
            kind=payload["typ"],
            audience=str(payload["aud"]),
            run_id=str(payload["run_id"]),
            tenant_id=str(payload["tenant_id"]),
            user_id=str(payload["user_id"]),
            issued_at=int(payload["iat"]),
            expires_at=int(payload["exp"]),
            token_id=str(payload["jti"]),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class TokenManager:
    secret: bytes
    issuer: str = TOKEN_ISSUER
    accepted_issuers: tuple[str, ...] = (LEGACY_TOKEN_ISSUER,)
    key_id: str = "default"
    verify_keys: Mapping[str, bytes] = field(default_factory=dict)
    retired_key_accept_until: Mapping[str, int] = field(default_factory=dict)
    revoked_token_ids: frozenset[str] = field(default_factory=frozenset)
    revoked_before: int = 0

    def __post_init__(self) -> None:
        secret = _coerce_secret(self.secret)
        key_id = str(self.key_id or "default")
        keys = {str(kid): _coerce_secret(value) for kid, value in self.verify_keys.items()}
        keys[key_id] = secret
        retired = {str(kid): int(until) for kid, until in self.retired_key_accept_until.items()}
        object.__setattr__(self, "secret", secret)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "verify_keys", keys)
        object.__setattr__(self, "retired_key_accept_until", retired)
        object.__setattr__(
            self,
            "revoked_token_ids",
            frozenset(str(token_id) for token_id in self.revoked_token_ids if str(token_id)),
        )
        object.__setattr__(self, "revoked_before", int(self.revoked_before or 0))

    @classmethod
    def ephemeral(cls) -> TokenManager:
        return cls(secrets.token_bytes(32))

    @classmethod
    def from_secret(cls, secret: str) -> TokenManager:
        if len(secret.encode("utf-8")) < 32:
            raise TokenError("token signing secret must be at least 32 bytes")
        return cls(secret.encode("utf-8"))

    @classmethod
    def from_keyring(
        cls,
        keys: Mapping[str, str | bytes],
        *,
        active_kid: str,
        issuer: str = TOKEN_ISSUER,
        accepted_issuers: tuple[str, ...] = (LEGACY_TOKEN_ISSUER,),
        retired_key_accept_until: Mapping[str, int] | None = None,
        revoked_token_ids: Iterable[str] = (),
        revoked_before: int = 0,
    ) -> TokenManager:
        keyring = {str(kid): _coerce_secret(secret) for kid, secret in keys.items()}
        if active_kid not in keyring:
            raise TokenError(f"active signing key not found: {active_kid}")
        return cls(
            secret=keyring[active_kid],
            issuer=issuer,
            accepted_issuers=accepted_issuers,
            key_id=active_kid,
            verify_keys=keyring,
            retired_key_accept_until=dict(retired_key_accept_until or {}),
            revoked_token_ids=frozenset(revoked_token_ids),
            revoked_before=revoked_before,
        )

    def rotate_key(self, *, key_id: str, secret: str | bytes, grace_s: int, now: int | None = None) -> TokenManager:
        current_time = int(time.time() if now is None else now)
        next_secret = _coerce_secret(secret)
        keys = {
            kid: value
            for kid, value in self.verify_keys.items()
            if self._key_accepted_for_verify(kid, current_time)
        }
        keys[str(key_id)] = next_secret
        retired = {
            kid: until
            for kid, until in self.retired_key_accept_until.items()
            if until >= current_time
        }
        if self.key_id != key_id:
            retired[self.key_id] = current_time + max(0, int(grace_s))
        return TokenManager(
            secret=next_secret,
            issuer=self.issuer,
            accepted_issuers=self.accepted_issuers,
            key_id=str(key_id),
            verify_keys=keys,
            retired_key_accept_until=retired,
            revoked_token_ids=self.revoked_token_ids,
            revoked_before=self.revoked_before,
        )

    def revoke_token_id(self, token_id: str) -> TokenManager:
        revoked = set(self.revoked_token_ids)
        revoked.add(str(token_id))
        return TokenManager(
            secret=self.secret,
            issuer=self.issuer,
            accepted_issuers=self.accepted_issuers,
            key_id=self.key_id,
            verify_keys=self.verify_keys,
            retired_key_accept_until=self.retired_key_accept_until,
            revoked_token_ids=frozenset(revoked),
            revoked_before=self.revoked_before,
        )

    def revoke_issued_before(self, before: int | float) -> TokenManager:
        return TokenManager(
            secret=self.secret,
            issuer=self.issuer,
            accepted_issuers=self.accepted_issuers,
            key_id=self.key_id,
            verify_keys=self.verify_keys,
            retired_key_accept_until=self.retired_key_accept_until,
            revoked_token_ids=self.revoked_token_ids,
            revoked_before=max(self.revoked_before, int(before)),
        )

    def issue(
        self,
        *,
        kind: TokenKind,
        audience: str,
        run_id: str,
        tenant_id: str,
        user_id: str,
        ttl_s: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = int(time.time())
        claims = TokenClaims(
            kind=kind,
            audience=audience,
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            issued_at=now,
            expires_at=now + ttl_s,
            metadata=dict(metadata or {}),
        )
        header = {"alg": "HS256", "typ": TOKEN_HEADER_TYPE, "kid": self.key_id}
        signing_input = ".".join(
            (
                _b64_json(header),
                _b64_json({"iss": self.issuer, **claims.to_json()}),
            )
        )
        signature = _b64_bytes(hmac.new(self.secret, signing_input.encode("utf-8"), hashlib.sha256).digest())
        return f"{signing_input}.{signature}"

    def verify(
        self,
        token: str,
        *,
        kind: TokenKind,
        audience: str | Iterable[str],
        run_id: str | None = None,
    ) -> TokenClaims:
        try:
            header_raw, payload_raw, signature = token.split(".", 2)
        except ValueError as exc:
            raise TokenError("invalid token format") from exc
        signing_input = f"{header_raw}.{payload_raw}"
        header = _json_b64(header_raw)
        if header.get("alg") != "HS256" or header.get("typ") not in ACCEPTED_TOKEN_HEADER_TYPES:
            raise TokenError("invalid token header")
        now = int(time.time())
        candidate_keys = self._candidate_keys_for_header(header, now)
        if not candidate_keys:
            raise TokenError("invalid or expired token signing key")
        if not any(
            hmac.compare_digest(
                signature,
                _b64_bytes(hmac.new(secret, signing_input.encode("utf-8"), hashlib.sha256).digest()),
            )
            for secret in candidate_keys
        ):
            raise TokenError("invalid token signature")
        payload = _json_b64(payload_raw)
        if payload.get("iss") not in (self.issuer, *self.accepted_issuers):
            raise TokenError("invalid token issuer")
        claims = TokenClaims.from_json(payload)
        if claims.kind != kind:
            raise TokenError("invalid token kind")
        if claims.audience not in normalize_audiences(audience):
            raise TokenError("invalid token audience")
        if run_id is not None and claims.run_id != run_id:
            raise TokenError("token run mismatch")
        if claims.expires_at < now:
            raise TokenError("token expired")
        if claims.token_id in self.revoked_token_ids or claims.issued_at < self.revoked_before:
            raise TokenError("token revoked")
        return claims

    def _candidate_keys_for_header(self, header: dict[str, Any], now: int) -> tuple[bytes, ...]:
        kid = header.get("kid")
        if kid:
            key_id = str(kid)
            if not self._key_accepted_for_verify(key_id, now):
                return ()
            key = self.verify_keys.get(key_id)
            return (key,) if key is not None else ()
        return tuple(
            secret
            for key_id, secret in self.verify_keys.items()
            if self._key_accepted_for_verify(key_id, now)
        )

    def _key_accepted_for_verify(self, key_id: str, now: int) -> bool:
        if key_id == self.key_id:
            return True
        accept_until = self.retired_key_accept_until.get(key_id)
        return accept_until is not None and now <= accept_until

    @staticmethod
    def token_sha256(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _coerce_secret(secret: str | bytes) -> bytes:
    value = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
    if len(value) < 32:
        raise TokenError("token signing secret must be at least 32 bytes")
    return value


def _b64_json(payload: dict[str, Any]) -> str:
    return _b64_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _b64_bytes(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _json_b64(payload: str) -> dict[str, Any]:
    padding = "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8"))
