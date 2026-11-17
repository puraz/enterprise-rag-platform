/**
 * 认证状态。
 *
 * ## 为什么不存"user 对象"而存"是否有 token"
 *
 * token 存在 localStorage 里（`tokenStore`），user 信息每次从
 * `/api/v1/me` 拉（TanStack Query 缓存）。
 *
 * 存 user 对象到 Zustand 会让"角色变了"需要手动同步——
 * 而角色变更必须**立即生效**（A7 的要求），所以必须以服务端为准。
 *
 * ## 为什么不校验 token 的 exp
 *
 * 前端解析 JWT 的 exp 需要读 payload，但：
 * - 它只是优化（省一次 401），不是安全边界（token 可被伪造）；
 * - 时钟偏差会让"看起来还没过期但服务端已拒"。
 *
 * 所以只判断"有没有 token"，过期交给 401 → 自动刷新 → 仍失败则清空。
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { create } from "zustand";
import type { CurrentUserResponse, LoginResponse } from "@contracts";
import { ApiError, apiFetch, tokenStore } from "@/lib/api";

interface AuthState {
  /** 是否有 access token（不代表有效，只代表"尝试过登录"）。 */
  hasToken: boolean;
  setHasToken: (value: boolean) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  hasToken: tokenStore.access !== null,
  setHasToken: (value) => set({ hasToken: value }),
}));

/** 当前用户信息（服务端为准）。 */
export function useCurrentUser() {
  const hasToken = useAuthStore((state) => state.hasToken);

  return useQuery<CurrentUserResponse, ApiError>({
    queryKey: ["auth", "me"],
    queryFn: () => apiFetch<CurrentUserResponse>("/api/v1/me"),
    enabled: hasToken,
    // 用户信息变化不频繁，但权限变更要尽快生效（A7 要求）
    staleTime: 60_000,
    retry: (failureCount, error) => {
      // 认证错误不重试（重试也没用）
      if (error.isAuthError) return false;
      return failureCount < 2;
    },
  });
}

/** dev 登录（本地开发用，生产环境后端不会注册该端点）。 */
export function useDevLogin() {
  const queryClient = useQueryClient();
  const setHasToken = useAuthStore((state) => state.setHasToken);

  return useMutation<LoginResponse, ApiError, { credential: string }>({
    // 契约字段是 `credential`（格式 `dev:<subject>`），不是 `subject`
    mutationFn: (payload) =>
      apiFetch<LoginResponse>("/api/v1/auth/dev-login", {
        method: "POST",
        body: payload,
      }),
    onSuccess: (data) => {
      tokenStore.set(data.access_token, data.refresh_token);
      setHasToken(true);
      // 登录后必须让 "me" 重新拉（旧的可能是 undefined 或别的用户）
      void queryClient.invalidateQueries({ queryKey: ["auth", "me"] });
    },
  });
}

/** 登出：清 token + 清缓存。 */
export function useLogout() {
  const queryClient = useQueryClient();
  const setHasToken = useAuthStore((state) => state.setHasToken);

  return () => {
    tokenStore.clear();
    setHasToken(false);
    // 必须清空全部缓存：不清的话下一个人登录会看到上一个人的数据
    queryClient.clear();
  };
}
