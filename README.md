# 我的 Fuwari 中文博客

这是一个基于 [Astro](https://astro.build/) 和 [Fuwari](https://github.com/saicaca/fuwari) 的中文个人博客，可免费部署到 GitHub Pages。

## 本地运行

需要 Node.js 20 或更高版本，以及 pnpm 9。

```bash
pnpm install
pnpm dev
```

浏览器打开 `http://localhost:4321`。

## 修改个人信息

编辑 `src/config.ts`：

- `siteConfig`：博客名称、简介、主题色和横幅
- `profileConfig`：头像、昵称和个人介绍
- `navBarConfig`：导航栏和 GitHub 地址

替换图片：

- 头像：`src/assets/images/demo-avatar.png`
- 横幅：`src/assets/images/demo-banner.png`

## 写文章

```bash
pnpm new-post 文章文件名
```

文章位于 `src/content/posts/`，使用 Markdown 编写。

## 发布到 GitHub Pages

1. 在 GitHub 创建一个公开仓库。
2. 将本项目推送到仓库的 `main` 分支。
3. 打开仓库的 `Settings → Pages`。
4. 在 `Build and deployment` 中将 Source 设为 `GitHub Actions`。
5. 等待 Actions 中的部署任务完成。

配置会自动适配以下两种地址：

- `用户名.github.io`
- `用户名.github.io/仓库名`

如果以后绑定独立域名，可通过 GitHub Actions 环境变量 `SITE_URL` 和 `BASE_PATH` 覆盖自动配置。
