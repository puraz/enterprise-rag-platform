/**
 * 登录页。
 *
 * ## 两种登录并存的判断
 *
 * 生产环境用 OIDC（跳转 IdP）；本地开发用 dev-login（直接输 subject）。
 *
 * 为什么不统一成 OIDC：本地没有 Keycloak（无 Docker），跑不起来。
 * 但**不能因此把 dev 登录当成唯一入口**——那是把开发后门带进生产。
 *
 * 做法：
 * - 默认展示 OIDC 入口（"使用企业账号登录"）；
 * - dev 表单只在 `import.meta.env.DEV` 下渲染（生产构建时被 tree-shake 掉）；
 * - 后端也会在非 local 环境**不注册** dev-login 端点（双重防护）。
 *
 * ## 视觉：不用居中大卡片 + 渐变背景
 *
 * 那是 SaaS 落地页的做法。内部工具应该克制：左侧一条窄的品牌区，
 * 右侧表单直接贴边。用户每天登录，花哨的背景只会碍事。
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useDevLogin } from "./useAuth";

export function LoginPage() {
  const navigate = useNavigate();
  const devLogin = useDevLogin();
  const [subject, setSubject] = useState("dev:alice@example.com");

  return (
    <div className="flex min-h-screen bg-white">
      {/* 左侧品牌区：深色窄栏，不用渐变 */}
      <div className="hidden w-80 shrink-0 flex-col justify-between bg-ink-950 p-8 md:flex">
        <span className="text-[13px] font-medium tracking-wide text-white">
          知识助手
        </span>
        <div>
          <p className="text-[15px] leading-relaxed text-ink-200">
            企业知识库问答。
            <br />
            每个回答都带来源引用。
          </p>
          <p className="mt-4 text-[12px] text-ink-500">
            多租户隔离 · 密级管控 · 全链路审计
          </p>
        </div>
      </div>

      {/* 右侧表单区 */}
      <div className="flex flex-1 items-center justify-center px-6">
        <div className="w-full max-w-sm">
          <h1 className="text-[18px] font-medium text-ink-900">登录</h1>
          <p className="mt-1 text-[13px] text-ink-500">
            使用企业账号继续。首次登录会自动开通账号。
          </p>

          <a
            href="/api/v1/auth/oidc/login"
            className="btn-primary mt-6 w-full"
          >
            使用企业账号登录
          </a>

          {/* dev 登录：仅开发环境渲染 */}
          {import.meta.env.DEV && (
            <div className="mt-8 border-t border-ink-200 pt-6">
              <p className="text-[12px] font-medium text-ink-600">
                本地开发登录
              </p>
              <p className="mt-1 text-[12px] text-ink-400">
                生产构建不会包含此表单，后端也不会注册该端点。
              </p>

              <div className="mt-3 flex gap-2">
                <input
                  className="input flex-1"
                  value={subject}
                  placeholder="dev:alice@example.com"
                  onChange={(event) => setSubject(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      devLogin.mutate(
                        { credential: subject },
                        { onSuccess: () => navigate("/chat") },
                      );
                    }
                  }}
                />
                <button
                  type="button"
                  className="btn-secondary"
                  disabled={devLogin.isPending || subject.trim().length === 0}
                  onClick={() =>
                    devLogin.mutate(
                      { credential: subject },
                      { onSuccess: () => navigate("/chat") },
                    )
                  }
                >
                  {devLogin.isPending ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    "登录"
                  )}
                </button>
              </div>

              {devLogin.error && (
                <p className="mt-2 text-[12px] text-danger">
                  {devLogin.error.message}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
