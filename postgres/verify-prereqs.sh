#!/usr/bin/env bash
set -euo pipefail

# verify-prereqs.sh
# Read-only preflight checks for the secure Postgres (RDS) access model.
# This script:
# - derives required identifiers dynamically
# - exports variables for later steps (in the current shell only if sourced)
# - fails fast with explicit, security-relevant explanations

# -----------------------------
# Usage
# -----------------------------
# Recommended:
#   source postgres/verify-prereqs.sh
#
# If you run it (not sourced), exported variables will not persist:
#   bash postgres/verify-prereqs.sh
#
# Optional overrides (env vars):
#   AWS_REGION (else uses aws configure)
#   CLUSTER_NAME, SERVICE_NAME, DB_INSTANCE_ID, POSTGRES_SECRET_NAME
#
# Notes:
# - Requires AWS CLI to be configured/authenticated for the target account.
# - All AWS calls below are read-only.

# -----------------------------
# Tooling requirements
# -----------------------------
need_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "ERROR: Required tool not found: $cmd"
    exit 1
  }
}

need_cmd aws
need_cmd jq

# -----------------------------
# 1) AWS identity and region
# -----------------------------
export AWS_REGION="${AWS_REGION:-$(aws configure get region)}"
if [ -z "${AWS_REGION}" ]; then
  echo "ERROR: AWS region is not configured. Set AWS_REGION or run: aws configure"
  exit 1
fi

export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
if [ -z "${ACCOUNT_ID}" ] || [ "${ACCOUNT_ID}" = "None" ]; then
  echo "ERROR: AWS CLI is not authenticated (sts get-caller-identity failed)."
  exit 1
fi

echo "Using AWS account ${ACCOUNT_ID} in region ${AWS_REGION}"
echo "Note: Variables exported by this script are intended for subsequent steps (persist only if sourced)."

# -----------------------------
# 2) ECS cluster and service
# -----------------------------
export CLUSTER_NAME="${CLUSTER_NAME:-smoke-cluster}"
export SERVICE_NAME="${SERVICE_NAME:-smoke-web-service-8080}"

CLUSTER_STATUS="$(aws ecs describe-clusters \
  --region "$AWS_REGION" \
  --clusters "$CLUSTER_NAME" \
  --query "clusters[0].status" \
  --output text 2>/dev/null || true)"

if [ "${CLUSTER_STATUS}" != "ACTIVE" ]; then
  echo "ERROR: ECS cluster ${CLUSTER_NAME} is not ACTIVE (status=${CLUSTER_STATUS})."
  echo "Failure means: expected ECS environment does not exist or is in a bad state."
  exit 1
fi

SERVICE_JSON="$(aws ecs describe-services \
  --region "$AWS_REGION" \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME" \
  --output json 2>/dev/null || true)"

SERVICE_STATUS="$(echo "$SERVICE_JSON" | jq -r '.services[0].status // empty')"
if [ -z "${SERVICE_STATUS}" ] || [ "${SERVICE_STATUS}" = "null" ]; then
  echo "ERROR: ECS service ${SERVICE_NAME} not found in cluster ${CLUSTER_NAME}."
  echo "Failure means: service is misnamed, deleted, or you are in the wrong account/region."
  exit 1
fi

echo "ECS cluster is ACTIVE; service exists: ${SERVICE_NAME}"

# -----------------------------
# 3) VPC and private subnets
# -----------------------------
export SUBNET_IDS="$(echo "$SERVICE_JSON" | jq -r '.services[0].networkConfiguration.awsvpcConfiguration.subnets[]?')"
if [ -z "${SUBNET_IDS}" ]; then
  echo "ERROR: No subnets found on ECS service network configuration."
  echo "Failure means: service is not using awsvpcConfiguration or has no subnets configured."
  exit 1
fi

FIRST_SUBNET="$(echo "$SUBNET_IDS" | head -n1)"
export VPC_ID="$(aws ec2 describe-subnets \
  --region "$AWS_REGION" \
  --subnet-ids "$FIRST_SUBNET" \
  --query "Subnets[0].VpcId" \
  --output text 2>/dev/null || true)"

if [ -z "${VPC_ID}" ] || [ "${VPC_ID}" = "None" ]; then
  echo "ERROR: Could not derive VPC ID from subnet ${FIRST_SUBNET}."
  exit 1
fi

for subnet in $SUBNET_IDS; do
  MAP_PUBLIC="$(aws ec2 describe-subnets \
    --region "$AWS_REGION" \
    --subnet-ids "$subnet" \
    --query "Subnets[0].MapPublicIpOnLaunch" \
    --output text 2>/dev/null || true)"

  if [ "${MAP_PUBLIC}" != "False" ]; then
    echo "ERROR: Subnet ${subnet} assigns public IPs (MapPublicIpOnLaunch=${MAP_PUBLIC})."
    echo "Failure means: ECS is running in a public subnet, violating private-only access."
    exit 1
  fi
done

echo "Validated private subnets:"
echo "$SUBNET_IDS" | sed 's/^/  - /'
echo "VPC_ID=${VPC_ID}"

# -----------------------------
# 4) Required VPC endpoints
# -----------------------------
# For private subnets with assignPublicIp=DISABLED, these endpoints allow:
# - pulling images from ECR
# - writing logs to CloudWatch
# - ECS Exec via SSM
# - ECS API connectivity (in private-only networks)
REQUIRED_ENDPOINTS=(
  s3
  ecr.api
  ecr.dkr
  logs
  ecs
  ssm
  ssmmessages
  ec2messages
)

ENDPOINTS="$(aws ec2 describe-vpc-endpoints \
  --region "$AWS_REGION" \
  --filters Name=vpc-id,Values="$VPC_ID" \
  --query "VpcEndpoints[].ServiceName" \
  --output text 2>/dev/null || true)"

if [ -z "${ENDPOINTS}" ]; then
  echo "ERROR: No VPC endpoints found for VPC ${VPC_ID}."
  echo "Failure means: private-subnet tasks will likely require NAT/internet or will fail (ECR/logs/Exec)."
  exit 1
fi

for ep in "${REQUIRED_ENDPOINTS[@]}"; do
  echo "${ENDPOINTS}" | grep -q "${ep}" || {
    echo "ERROR: Missing required VPC endpoint containing: ${ep}"
    echo "Failure means: private-subnet tasks may fail to pull images, write logs, or use ECS Exec."
    exit 1
  }
done

echo "All required VPC endpoints are present."

# -----------------------------
# 5) RDS Postgres instance
# -----------------------------
export DB_INSTANCE_ID="${DB_INSTANCE_ID:-llm-code-postgres}"

DB_JSON="$(aws rds describe-db-instances \
  --region "$AWS_REGION" \
  --db-instance-identifier "$DB_INSTANCE_ID" \
  --output json 2>/dev/null || true)"

DB_STATUS="$(echo "$DB_JSON" | jq -r '.DBInstances[0].DBInstanceStatus // empty')"
if [ "${DB_STATUS}" != "available" ]; then
  echo "ERROR: RDS instance ${DB_INSTANCE_ID} is not available (status=${DB_STATUS:-unknown})."
  echo "Failure means: database is stopped/modifying/failed; connections will fail safely."
  exit 1
fi

export RDS_ENDPOINT="$(echo "$DB_JSON" | jq -r '.DBInstances[0].Endpoint.Address // empty')"
if [ -z "${RDS_ENDPOINT}" ] || [ "${RDS_ENDPOINT}" = "null" ]; then
  echo "ERROR: Could not read RDS endpoint for ${DB_INSTANCE_ID}."
  exit 1
fi

echo "RDS instance is available; endpoint: ${RDS_ENDPOINT}"

# -----------------------------
# 6) Secrets Manager (runtime credentials)
# -----------------------------
export POSTGRES_SECRET_NAME="${POSTGRES_SECRET_NAME:-llm-code/postgres-uri}"

export POSTGRES_SECRET_ARN="$(aws secretsmanager describe-secret \
  --region "$AWS_REGION" \
  --secret-id "$POSTGRES_SECRET_NAME" \
  --query ARN \
  --output text 2>/dev/null || true)"

if [ -z "${POSTGRES_SECRET_ARN}" ] || [ "${POSTGRES_SECRET_ARN}" = "None" ]; then
  echo "ERROR: Secrets Manager secret not found: ${POSTGRES_SECRET_NAME}"
  echo "Failure means: runtime credentials are not centrally managed or name is wrong."
  exit 1
fi

echo "Secrets Manager secret found (credentials stored in AWS): ${POSTGRES_SECRET_ARN}"

# -----------------------------
# Summary (safe to paste into tickets)
# -----------------------------
cat <<EOF

OK: prereqs satisfied.

Exported variables:
- AWS_REGION=${AWS_REGION}
- ACCOUNT_ID=${ACCOUNT_ID}
- CLUSTER_NAME=${CLUSTER_NAME}
- SERVICE_NAME=${SERVICE_NAME}
- VPC_ID=${VPC_ID}
- SUBNET_IDS=$(echo "$SUBNET_IDS" | tr '\n' ' ' | sed 's/ *$//')
- DB_INSTANCE_ID=${DB_INSTANCE_ID}
- RDS_ENDPOINT=${RDS_ENDPOINT}
- POSTGRES_SECRET_NAME=${POSTGRES_SECRET_NAME}
- POSTGRES_SECRET_ARN=${POSTGRES_SECRET_ARN}

Next:
- build/push db-tools image to ECR using ACCOUNT_ID/AWS_REGION
- run one-off db-tools ECS task in SUBNET_IDS (private) with assignPublicIp=DISABLED
- ECS Exec into task for controlled admin actions, then stop the task
EOF
