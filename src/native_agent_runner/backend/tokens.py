from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from native_agent_runner.errors import NativeAgentError

TokenKind = Literal["run_access", "llm_gateway", "web_gateway"]


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
    issuer: str = "native-agent-runner"

    @classmethod
    def ephemeral(cls) -> TokenManager:
        return cls(secrets.token_bytes(32))

    @classmethod
    def from_secret(cls, secret: str) -> TokenManager:
        if len(secret.encode("utf-8")) < 32:
            raise TokenError("token signing secret must be at least 32 bytes")
        return cls(secret.encode("utf-8"))

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
        header = {"alg": "HS256", "typ": "NAR"}
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
        audience: str,
        run_id: str | None = None,
    ) -> TokenClaims:
        try:
            header_raw, payload_raw, signature = token.split(".", 2)
        except ValueError as exc:
            raise TokenError("invalid token format") from exc
        signing_input = f"{header_raw}.{payload_raw}"
        expected = _b64_bytes(hmac.new(self.secret, signing_input.encode("utf-8"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise TokenError("invalid token signature")
        header = _json_b64(header_raw)
        if header.get("alg") != "HS256" or header.get("typ") != "NAR":
            raise TokenError("invalid token header")
        payload = _json_b64(payload_raw)
        if payload.get("iss") != self.issuer:
            raise TokenError("invalid token issuer")
        claims = TokenClaims.from_json(payload)
        if claims.kind != kind:
            raise TokenError("invalid token kind")
        if claims.audience != audience:
            raise TokenError("invalid token audience")
        if run_id is not None and claims.run_id != run_id:
            raise TokenError("token run mismatch")
        if claims.expires_at < int(time.time()):
            raise TokenError("token expired")
        return claims

    @staticmethod
    def token_sha256(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _b64_json(payload: dict[str, Any]) -> str:
    return _b64_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _b64_bytes(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _json_b64(payload: str) -> dict[str, Any]:
    padding = "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8"))
