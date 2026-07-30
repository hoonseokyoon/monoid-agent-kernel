from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import secrets
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from monoid_agent_kernel.core.json_ingress import (
    loads_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import LEGACY_TOKEN_ISSUER, TOKEN_ISSUER, normalize_audiences

TokenKind = Literal["run_access", "llm_gateway", "web_gateway", "task_callback", "capability"]
TokenTextCollection = list[str] | tuple[str, ...] | set[str] | frozenset[str]
TOKEN_HEADER_TYPE = "MAK"
LEGACY_TOKEN_HEADER_TYPE = "NAR"
ACCEPTED_TOKEN_HEADER_TYPES = (TOKEN_HEADER_TYPE, LEGACY_TOKEN_HEADER_TYPE)


class TokenError(NativeAgentError):
    pass


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TokenError(f"{field_name} must be a non-empty string")
    return normalize_unicode_scalars(value)


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise TokenError(f"{field_name} must be a non-negative integer")
    return value


def _finite_nonnegative_number(value: Any, field_name: str) -> int | float:
    if type(value) is int and value >= 0:
        return value
    if type(value) is float and math.isfinite(value) and value >= 0:
        return value
    raise TokenError(f"{field_name} must be a finite non-negative number")


def _text_collection(value: Any, field_name: str) -> tuple[str, ...]:
    if type(value) not in {list, tuple, set, frozenset}:
        raise TokenError(f"{field_name} must be a list, tuple, set, or frozenset of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if type(item) is not str or not item:
            raise TokenError(f"{field_name} entries must be non-empty strings")
        text = normalize_unicode_scalars(item)
        if text in seen:
            raise TokenError(f"{field_name} entries collide after ingress normalization")
        seen.add(text)
        normalized.append(text)
    if type(value) in {set, frozenset}:
        normalized.sort()
    return tuple(normalized)


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
        if self.kind not in {"run_access", "llm_gateway", "web_gateway", "task_callback", "capability"}:
            raise TokenError("invalid token kind")
        metadata = normalize_json_ingress(self.metadata)
        if not isinstance(metadata, dict):
            raise TokenError("token metadata must be an object")
        return {
            "typ": self.kind,
            "aud": _required_text(self.audience, "token audience"),
            "run_id": _required_text(self.run_id, "token run_id"),
            "tenant_id": _required_text(self.tenant_id, "token tenant_id"),
            "user_id": _required_text(self.user_id, "token user_id"),
            "iat": _nonnegative_integer(self.issued_at, "token issued_at"),
            "exp": _nonnegative_integer(self.expires_at, "token expires_at"),
            "jti": _required_text(self.token_id, "token id"),
            "metadata": metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TokenClaims:
        payload = normalize_json_ingress(payload)
        if not isinstance(payload, dict):
            raise TokenError("token claims must be an object")
        kind = _required_text(payload.get("typ"), "token kind")
        if kind not in {"run_access", "llm_gateway", "web_gateway", "task_callback", "capability"}:
            raise TokenError("invalid token kind")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TokenError("token metadata must be an object")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            audience=_required_text(payload.get("aud"), "token audience"),
            run_id=_required_text(payload.get("run_id"), "token run_id"),
            tenant_id=_required_text(payload.get("tenant_id"), "token tenant_id"),
            user_id=_required_text(payload.get("user_id"), "token user_id"),
            issued_at=_nonnegative_integer(payload.get("iat"), "token issued_at"),
            expires_at=_nonnegative_integer(payload.get("exp"), "token expires_at"),
            token_id=_required_text(payload.get("jti"), "token id"),
            metadata=metadata,
        )


@dataclass(frozen=True)
class TokenManager:
    secret: bytes
    issuer: str = TOKEN_ISSUER
    accepted_issuers: TokenTextCollection = (LEGACY_TOKEN_ISSUER,)
    key_id: str = "default"
    verify_keys: Mapping[str, bytes] = field(default_factory=dict)
    retired_key_accept_until: Mapping[str, int] = field(default_factory=dict)
    revoked_token_ids: TokenTextCollection = field(default_factory=frozenset)
    revoked_before: int = 0

    def __post_init__(self) -> None:
        secret = _coerce_secret(self.secret)
        key_id = _required_text(self.key_id, "active token key id")
        keys: dict[str, bytes] = {}
        for raw_key_id, value in self.verify_keys.items():
            normalized_key_id = _required_text(raw_key_id, "token key id")
            if normalized_key_id in keys:
                raise TokenError("token key ids collide after ingress normalization")
            if normalized_key_id == key_id and raw_key_id != self.key_id:
                raise TokenError("token key ids collide after ingress normalization")
            keys[normalized_key_id] = _coerce_secret(value)
        keys[key_id] = secret
        retired: dict[str, int] = {}
        for raw_key_id, until in self.retired_key_accept_until.items():
            normalized_key_id = _required_text(raw_key_id, "retired token key id")
            if normalized_key_id in retired:
                raise TokenError("retired token key ids collide after ingress normalization")
            retired[normalized_key_id] = _nonnegative_integer(
                until,
                "retired token key accept-until epoch",
            )
        object.__setattr__(self, "secret", secret)
        object.__setattr__(self, "issuer", _required_text(self.issuer, "token issuer"))
        object.__setattr__(
            self,
            "accepted_issuers",
            _text_collection(self.accepted_issuers, "accepted token issuers"),
        )
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "verify_keys", keys)
        object.__setattr__(self, "retired_key_accept_until", retired)
        object.__setattr__(
            self,
            "revoked_token_ids",
            frozenset(_text_collection(self.revoked_token_ids, "revoked token ids")),
        )
        object.__setattr__(
            self,
            "revoked_before",
            _nonnegative_integer(self.revoked_before, "token revoked-before epoch"),
        )

    @classmethod
    def ephemeral(cls) -> TokenManager:
        return cls(secrets.token_bytes(32))

    @classmethod
    def from_secret(cls, secret: str) -> TokenManager:
        return cls(_coerce_secret(secret))

    @classmethod
    def from_keyring(
        cls,
        keys: Mapping[str, str | bytes],
        *,
        active_kid: str,
        issuer: str = TOKEN_ISSUER,
        accepted_issuers: TokenTextCollection = (LEGACY_TOKEN_ISSUER,),
        retired_key_accept_until: Mapping[str, int] | None = None,
        revoked_token_ids: TokenTextCollection = (),
        revoked_before: int = 0,
    ) -> TokenManager:
        active_key_id = _required_text(active_kid, "active token key id")
        keyring: dict[str, bytes] = {}
        for raw_key_id, secret in keys.items():
            normalized_key_id = _required_text(raw_key_id, "token key id")
            if normalized_key_id in keyring:
                raise TokenError("token key ids collide after ingress normalization")
            keyring[normalized_key_id] = _coerce_secret(secret)
        if active_key_id not in keyring:
            raise TokenError(f"active signing key not found: {active_key_id}")
        return cls(
            secret=keyring[active_key_id],
            issuer=issuer,
            accepted_issuers=accepted_issuers,
            key_id=active_key_id,
            verify_keys=keyring,
            retired_key_accept_until=dict(retired_key_accept_until or {}),
            revoked_token_ids=revoked_token_ids,
            revoked_before=revoked_before,
        )

    def rotate_key(self, *, key_id: str, secret: str | bytes, grace_s: int, now: int | None = None) -> TokenManager:
        current_time = (
            int(time.time())
            if now is None
            else _nonnegative_integer(now, "token key rotation now")
        )
        grace_seconds = _nonnegative_integer(grace_s, "token key rotation grace_s")
        next_key_id = _required_text(key_id, "token key rotation key id")
        next_secret = _coerce_secret(secret)
        keys = {
            kid: value
            for kid, value in self.verify_keys.items()
            if self._key_accepted_for_verify(kid, current_time)
        }
        keys[next_key_id] = next_secret
        retired = {
            kid: until
            for kid, until in self.retired_key_accept_until.items()
            if until >= current_time
        }
        if self.key_id != next_key_id:
            retired[self.key_id] = current_time + grace_seconds
        return TokenManager(
            secret=next_secret,
            issuer=self.issuer,
            accepted_issuers=self.accepted_issuers,
            key_id=next_key_id,
            verify_keys=keys,
            retired_key_accept_until=retired,
            revoked_token_ids=self.revoked_token_ids,
            revoked_before=self.revoked_before,
        )

    def revoke_token_id(self, token_id: str) -> TokenManager:
        revoked = set(self.revoked_token_ids)
        revoked.add(_required_text(token_id, "revoked token id"))
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
            revoked_before=max(self.revoked_before, _ceil_epoch(before)),
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
        if type(ttl_s) is not int or ttl_s <= 0:
            raise TokenError("token ttl_s must be a positive integer")
        if metadata is not None and not isinstance(metadata, dict):
            raise TokenError("token metadata must be an object or null")
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
        expected_kind = _required_text(kind, "expected token kind")
        expected_audiences = tuple(
            _required_text(value, "expected token audience")
            for value in normalize_audiences(audience)
        )
        expected_run_id = (
            _required_text(run_id, "expected token run_id") if run_id is not None else None
        )
        try:
            header_raw, payload_raw, signature = token.split(".", 2)
        except ValueError as exc:
            raise TokenError("invalid token format") from exc
        signing_input = f"{header_raw}.{payload_raw}"
        header = _token_json_b64(header_raw, "header")
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
        payload = _token_json_b64(payload_raw, "payload")
        if payload.get("iss") not in (self.issuer, *self.accepted_issuers):
            raise TokenError("invalid token issuer")
        claims = TokenClaims.from_json(payload)
        if claims.kind != expected_kind:
            raise TokenError("invalid token kind")
        if claims.audience not in expected_audiences:
            raise TokenError("invalid token audience")
        if expected_run_id is not None and claims.run_id != expected_run_id:
            raise TokenError("token run mismatch")
        if claims.expires_at < now:
            raise TokenError("token expired")
        if claims.token_id in self.revoked_token_ids or claims.issued_at < self.revoked_before:
            raise TokenError("token revoked")
        return claims

    def _candidate_keys_for_header(self, header: dict[str, Any], now: int) -> tuple[bytes, ...]:
        if "kid" in header:
            key_id = _required_text(header["kid"], "token header kid")
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


def _coerce_secret(secret: Any) -> bytes:
    if type(secret) is str:
        try:
            value = secret.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise TokenError("token signing secret must contain valid Unicode scalar values") from exc
    elif type(secret) is bytes:
        value = secret
    else:
        raise TokenError("token signing secret must be exactly str or bytes")
    if len(value) < 32:
        raise TokenError("token signing secret must be at least 32 bytes")
    return value


def _ceil_epoch(value: int | float) -> int:
    return int(math.ceil(_finite_nonnegative_number(value, "token revocation cutoff")))


def _token_json_b64(value: str, label: str) -> dict[str, Any]:
    try:
        return _json_b64(value)
    except (binascii.Error, TypeError, ValueError) as exc:
        raise TokenError(f"invalid token {label}") from exc


def _b64_json(payload: dict[str, Any]) -> str:
    normalized = normalize_json_ingress(payload)
    if not isinstance(normalized, dict):
        raise TokenError("token JSON payload must be an object")
    return _b64_bytes(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _b64_bytes(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _json_b64(payload: str) -> dict[str, Any]:
    padding = "=" * (-len(payload) % 4)
    decoded = loads_json_ingress(
        base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8")
    )
    if not isinstance(decoded, dict):
        raise ValueError("token JSON payload must be an object")
    return decoded
