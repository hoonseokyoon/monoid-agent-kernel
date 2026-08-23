# v0.23 actual-service harness

이 harness는 repository-owned PostgreSQL·MinIO container와 Temporal local server를 실행한다.
`campaign-lock.json`이 SDK exact version, image digest, Temporal CLI archive checksum을 소유한다.

## local combined gate

PowerShell에서 다음 순서로 실행한다.

```powershell
$env:COMPOSE_PROFILES = "combined"
$env:MONOID_SERVICE_PROFILE = "combined"
$env:MONOID_POSTGRES16_DSN = "postgresql://monoid:monoid-local-only@127.0.0.1:55416/monoid_kernel_test"
$env:MONOID_POSTGRES18_DSN = "postgresql://monoid:monoid-local-only@127.0.0.1:55418/monoid_kernel_test"
$env:MONOID_MINIO_ENDPOINT = "http://127.0.0.1:59000"
$env:MONOID_MINIO_ACCESS_KEY = "monoid-test-access"
$env:MONOID_MINIO_SECRET_KEY = "monoid-test-secret-only"
$env:MONOID_TEMPORAL_CLI_VERSION = "v1.8.2"

python -m pip install -e ".[dev,durable-host]"
python tools/v023_ci.py validate-lock
$campaignRequirements = (python tools/v023_ci.py exact-requirements) -split " "
python -m pip install @campaignRequirements
python tools/v023_ci.py verify-installed
docker compose -f tests/service/compose.yml up -d --wait
python -m pytest -q tests/service -m service
docker compose -f tests/service/compose.yml down --volumes
```

고정 credential은 local service 전용이다. production configuration에 사용하지 않는다.

## profiles

| profile | services |
|---|---|
| `postgres` | PostgreSQL 16 |
| `objectstore` | PostgreSQL 16 + MinIO |
| `temporal` | Temporal CLI local server |
| `combined` | PostgreSQL 16/18 + MinIO + Temporal CLI local server |
| `release` | PostgreSQL 16/18 + MinIO container matrix |

Temporal은 test process가 `WorkflowEnvironment.start_local()`로 시작하고 종료한다. Docker Compose는
PostgreSQL과 MinIO만 소유한다. Temporal test는 현재 OS/architecture의 공식 CLI archive를
campaign lock SHA-256과 비교하고, 검증한 archive에서 추출한 executable을 local server로 실행한다.
