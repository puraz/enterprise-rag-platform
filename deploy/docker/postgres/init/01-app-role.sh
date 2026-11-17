#!/bin/sh
# PostgreSQL 首次初始化：创建应用角色 + pgvector 扩展。
#
# ## 为什么需要这个脚本
#
# 官方 postgres 镜像只创建一个 `POSTGRES_USER`（这里是 rag_admin，开发环境的
# 迁移/管理员角色）。但应用运行时必须用**受 RLS 约束**的普通角色 `rag_app`：
# - 用超级用户跑应用会让 RLS 完全失效（超级用户无条件绕过策略）；
# - worker 的跨租户认领需要 BYPASSRLS 的角色（这里暂时复用 rag_admin）。
#
# 本脚本只在**数据卷为空**时执行一次（postgres 镜像的 initdb 语义）。
# 生产环境应由 DBA / 密钥管理系统创建角色，不要依赖这个开发脚本。
#
# ## 密码来自环境变量
#
# `RAG_APP_PASSWORD` 由 compose 注入。默认值只用于本地开发；
# 生产必须覆盖，且不要用特殊字符（psql 变量会按 SQL 字面量处理单引号，
# 但 shell 层仍受引号影响）。
set -eu

APP_ROLE="${RAG_APP_ROLE:-rag_app}"
APP_PASSWORD="${RAG_APP_PASSWORD:?RAG_APP_PASSWORD must be set}"

echo "[init] creating role ${APP_ROLE} and enabling pgvector"

if ! psql -tAc "SELECT 1 FROM pg_roles WHERE rolname = '${APP_ROLE}'" \
        --username "$POSTGRES_USER" \
        --dbname "$POSTGRES_DB" | grep -q 1; then
    # 密码通过 psql 变量 + `:'app_password'` 按 SQL 字面量引用。
    # 注意：psql 变量插值只在从 stdin/文件读取命令时生效，
    # `psql -c "... :'var' ..."` 不会展开（实测会留下冒号导致语法错误）。
    psql -v ON_ERROR_STOP=1 \
        --username "$POSTGRES_USER" \
        --dbname "$POSTGRES_DB" \
        -v app_password="$APP_PASSWORD" <<SQL
CREATE ROLE ${APP_ROLE} LOGIN PASSWORD :'app_password' NOSUPERUSER NOBYPASSRLS;
SQL
fi

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" <<'SQL'
-- 向量检索的基础扩展。迁移里也有 CREATE EXTENSION IF NOT EXISTS，
-- 这里提前建好可以避免迁移角色权限不足时卡在第一步。
CREATE EXTENSION IF NOT EXISTS vector;
SQL

# 数据库级连接权限 + schema 使用权限。
# 表级 GRANT 由 Alembic 迁移显式执行（每个新表都必须授权，这是有意设计：
# 忘了授权会在集成测里立刻暴露，而不是“生产上某个角色看不到表”）。
psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -c "GRANT CONNECT ON DATABASE \"${POSTGRES_DB}\" TO ${APP_ROLE};"
psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -c "GRANT USAGE ON SCHEMA public TO ${APP_ROLE};"

# 为迁移角色后续新建的表预授权（迁移本身也会显式 GRANT；
# 这里只是减少“新表忘授权”的人为失误）。
psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -c "ALTER DEFAULT PRIVILEGES FOR ROLE \"${POSTGRES_USER}\" IN SCHEMA public \
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${APP_ROLE};"
psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -c "ALTER DEFAULT PRIVILEGES FOR ROLE \"${POSTGRES_USER}\" IN SCHEMA public \
        GRANT USAGE, SELECT ON SEQUENCES TO ${APP_ROLE};"

echo "[init] role ${APP_ROLE} ready"
