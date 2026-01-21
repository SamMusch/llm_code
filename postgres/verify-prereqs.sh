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
#   ENFORCE_PRIVATE_SUBNETS=1 (default: 0) to fail if subnets are public
#   INCLUDE_EXTENDED_AWS=1 (default: 1) to include ECS/ALB/ECR/CloudWatch inventory
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

# Extended inventory toggle (read-only)
export INCLUDE_EXTENDED_AWS="${INCLUDE_EXTENDED_AWS:-1}"

# Service networking facts (descriptive)
export ASSIGN_PUBLIC_IP="$(echo "$SERVICE_JSON" | jq -r '.services[0].networkConfiguration.awsvpcConfiguration.assignPublicIp // empty')"
export SERVICE_SECURITY_GROUP_IDS="$(echo "$SERVICE_JSON" | jq -r '.services[0].networkConfiguration.awsvpcConfiguration.securityGroups[]?')"
if [ -n "${ASSIGN_PUBLIC_IP}" ]; then
  echo "ECS service assignPublicIp=${ASSIGN_PUBLIC_IP}"
fi
if [ -n "${SERVICE_SECURITY_GROUP_IDS}" ]; then
  echo "ECS service security groups:"
  echo "$SERVICE_SECURITY_GROUP_IDS" | sed 's/^/  - /'
fi

# By default this script is used for audit/read-only inventory and will not fail on public subnet settings.
# Set ENFORCE_PRIVATE_SUBNETS=1 to require private subnets (MapPublicIpOnLaunch=False) and fail if violated.
export ENFORCE_PRIVATE_SUBNETS="${ENFORCE_PRIVATE_SUBNETS:-0}"

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
    if [ "${ENFORCE_PRIVATE_SUBNETS}" = "1" ]; then
      echo "ERROR: Subnet ${subnet} assigns public IPs (MapPublicIpOnLaunch=${MAP_PUBLIC})."
      echo "Failure means: ECS is running in a public subnet, violating private-only access."
      exit 1
    else
      echo "WARN: Subnet ${subnet} assigns public IPs (MapPublicIpOnLaunch=${MAP_PUBLIC})."
      echo "WARN: Audit mode continues. Private-only access is NOT enforced unless ENFORCE_PRIVATE_SUBNETS=1."
    fi
  fi
done

echo "Subnets (private-only enforced: ${ENFORCE_PRIVATE_SUBNETS}):"
echo "$SUBNET_IDS" | sed 's/^/  - /'
echo "VPC_ID=${VPC_ID}"

# -----------------------------
# 4) Required VPC endpoints
# -----------------------------
# These endpoints are commonly required when ECS tasks do not have direct internet egress.
# They support pulling images from ECR, writing logs to CloudWatch, and ECS Exec via SSM.
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

OK: audit checks completed.

Key facts:
- AWS_REGION=${AWS_REGION}
- ACCOUNT_ID=${ACCOUNT_ID}
- CLUSTER_NAME=${CLUSTER_NAME}
- SERVICE_NAME=${SERVICE_NAME}
- VPC_ID=${VPC_ID}
- SUBNET_IDS=$(echo "$SUBNET_IDS" | tr '\n' ' ' | sed 's/ *$//')
- ASSIGN_PUBLIC_IP=${ASSIGN_PUBLIC_IP:-unknown}
- SERVICE_SECURITY_GROUP_IDS=$(echo "$SERVICE_SECURITY_GROUP_IDS" | tr '\n' ' ' | sed 's/ *$//')
- DB_INSTANCE_ID=${DB_INSTANCE_ID}
- RDS_ENDPOINT=${RDS_ENDPOINT}
- POSTGRES_SECRET_NAME=${POSTGRES_SECRET_NAME}
- POSTGRES_SECRET_ARN=${POSTGRES_SECRET_ARN}
- PRIVATE_ONLY_ENFORCED=${ENFORCE_PRIVATE_SUBNETS}

EOF

# -----------------------------
# Extended AWS inventory (ECS/ALB/ECR/CloudWatch)
# -----------------------------
if [ "${INCLUDE_EXTENDED_AWS}" = "1" ]; then
  export TASK_DEFINITION_ARN="$(echo "$SERVICE_JSON" | jq -r '.services[0].taskDefinition // empty')"
  if [ -n "${TASK_DEFINITION_ARN}" ]; then
    echo "ECS task definition: ${TASK_DEFINITION_ARN}"

    TD_JSON="$(aws ecs describe-task-definition \
      --region "$AWS_REGION" \
      --task-definition "$TASK_DEFINITION_ARN" \
      --output json 2>/dev/null || true)"

    export TASK_DEFINITION_FAMILY="$(echo "$TD_JSON" | jq -r '.taskDefinition.family // empty')"
    export TASK_DEFINITION_REVISION="$(echo "$TD_JSON" | jq -r '.taskDefinition.revision // empty')"
    export CONTAINER_NAMES="$(echo "$TD_JSON" | jq -r '.taskDefinition.containerDefinitions[].name')"
    export CONTAINER_IMAGES="$(echo "$TD_JSON" | jq -r '.taskDefinition.containerDefinitions[].image')"
    export CLOUDWATCH_LOG_GROUPS="$(echo "$TD_JSON" | jq -r '.taskDefinition.containerDefinitions[].logConfiguration.options["awslogs-group"]?')"

    if [ -n "${TASK_DEFINITION_FAMILY}" ]; then
      echo "Task family: ${TASK_DEFINITION_FAMILY}"; fi
    if [ -n "${TASK_DEFINITION_REVISION}" ]; then
      echo "Task revision: ${TASK_DEFINITION_REVISION}"; fi

    if [ -n "${CONTAINER_NAMES}" ]; then
      echo "Task containers:"; echo "$CONTAINER_NAMES" | sed 's/^/  - /'; fi
    if [ -n "${CONTAINER_IMAGES}" ]; then
      echo "Task images:"; echo "$CONTAINER_IMAGES" | sed 's/^/  - /'; fi
    if [ -n "${CLOUDWATCH_LOG_GROUPS}" ]; then
      echo "CloudWatch log groups (from task definition):"; echo "$CLOUDWATCH_LOG_GROUPS" | sed 's/^/  - /'; fi

    # Derive ECR repository names from image URIs (best-effort)
    ECR_REPOS_DERIVED="$(echo "$CONTAINER_IMAGES" | sed -n 's#^[0-9]\{12\}\.dkr\.ecr\.[^/]\+\.amazonaws\.com/\([^:@]\+\).*#\1#p' | sort -u)"
    if [ -n "${ECR_REPOS_DERIVED}" ]; then
      echo "ECR repositories (derived from images):"; echo "$ECR_REPOS_DERIVED" | sed 's/^/  - /'
    fi
  fi

  # Load balancer / target group bindings (from ECS service)
  export TARGET_GROUP_ARNS="$(echo "$SERVICE_JSON" | jq -r '.services[0].loadBalancers[].targetGroupArn?')"
  if [ -n "${TARGET_GROUP_ARNS}" ]; then
    echo "Target groups (from ECS service):"; echo "$TARGET_GROUP_ARNS" | sed 's/^/  - /'

    # Attempt to resolve ALB ARNs and DNS names from target groups (best-effort)
    TG_JSON="$(aws elbv2 describe-target-groups \
      --region "$AWS_REGION" \
      --target-group-arns $(echo "$TARGET_GROUP_ARNS" | tr '\n' ' ') \
      --output json 2>/dev/null || true)"

    export LOAD_BALANCER_ARNS="$(echo "$TG_JSON" | jq -r '.TargetGroups[].LoadBalancerArns[]?' | sort -u)"
    if [ -n "${LOAD_BALANCER_ARNS}" ]; then
      echo "Load balancers (from target groups):"; echo "$LOAD_BALANCER_ARNS" | sed 's/^/  - /'

      LB_JSON="$(aws elbv2 describe-load-balancers \
        --region "$AWS_REGION" \
        --load-balancer-arns $(echo "$LOAD_BALANCER_ARNS" | tr '\n' ' ') \
        --output json 2>/dev/null || true)"

      export LOAD_BALANCER_NAMES="$(echo "$LB_JSON" | jq -r '.LoadBalancers[].LoadBalancerName?')"
      export LOAD_BALANCER_DNS_NAMES="$(echo "$LB_JSON" | jq -r '.LoadBalancers[].DNSName?')"
      if [ -n "${LOAD_BALANCER_NAMES}" ]; then
        echo "Load balancer names:"; echo "$LOAD_BALANCER_NAMES" | sed 's/^/  - /'; fi
      if [ -n "${LOAD_BALANCER_DNS_NAMES}" ]; then
        echo "Load balancer DNS names:"; echo "$LOAD_BALANCER_DNS_NAMES" | sed 's/^/  - /'; fi
    fi
  fi
fi
