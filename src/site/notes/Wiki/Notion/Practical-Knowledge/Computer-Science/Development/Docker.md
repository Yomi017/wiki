---
{"dg-publish":true,"permalink":"/wiki/notion/practical-knowledge/computer-science/development/docker/"}
---

# Docker 容器单独配置 DNS 修复

## 1. 什么时候要考虑给 Docker 容器单独配 DNS

在 WSL2 + Docker Desktop 或 Docker Engine 的环境里，容器默认会走 Docker 内置 DNS `127.0.0.11`。  
如果宿主机当前 DNS 链路里混入了 Tailscale、校园网、公司内网、代理软件等配置，容器就可能出现下面这种现象：

1. 宿主机能访问，容器不能访问。
2. 容器访问某些公网域名时卡死在 `lookup xxx: server misbehaving`。
3. `curl` / `wget` / 应用日志里表现为 DNS 解析失败，但代码本身没问题。
4. 同一台机器上，某些容器正常，某些容器异常。

这类问题很适合用“**只给指定容器单独配置公共 DNS**”的方法处理，而不是一上来就改整个 WSL 的 `/etc/resolv.conf`。

## 2. 这次实际踩到的问题

### ① WebCapture

`browserless`、`toolhub-api`、`toolhub-mcp` 之前会继承坏掉的宿主机 DNS 和搜索域，导致公网网页抓取失败。  
最后的修法是：

```yaml
x-webcapture-dns: &webcapture_dns
  dns:
    - "${TOOLHUB_WEBCAPTURE_DNS_PRIMARY:-223.5.5.5}"
    - "${TOOLHUB_WEBCAPTURE_DNS_SECONDARY:-119.29.29.29}"
  dns_search: []
```

然后把这组锚点挂到需要的容器上。

### ② sub2api

`sub2api` 里访问 `api.qinzhiai.com`、`chatgpt.com` 时，最开始一直报 DNS 解析失败。  
真正原因有两层：

1. 容器还在吃 Docker 默认 DNS，解析公网域名不稳定。
2. Compose 里写的是 `HTTP_PROXY=${HTTP_PROXY:-}`，结果被宿主机环境变量偷偷覆盖成了 `127.0.0.1:7890`，容器里根本连不到这个地址。

所以这次 `sub2api` 的修法不是只有 DNS，还顺手把代理变量名也收口了。

## 3. 推荐修法：容器独立 DNS + 专用环境变量

### ① DNS 独立配置

以 `sub2api` 为例，可以在 compose 顶部加一个锚点：

```yaml
x-sub2api-dns: &sub2api_dns
  dns:
    - "${SUB2API_DNS_PRIMARY:-223.5.5.5}"
    - "${SUB2API_DNS_SECONDARY:-119.29.29.29}"
  dns_search: []
```

然后挂到目标服务：

```yaml
services:
  sub2api:
    <<: *sub2api_dns
```

这里 `dns_search: []` 很重要。  
如果不清空搜索域，容器可能会继续继承类似 `tailca3707.ts.net` 这种搜索域，导致公网域名解析行为继续异常。

### ② 不要直接读全局 `HTTP_PROXY`

如果 compose 里直接写：

```yaml
- HTTP_PROXY=${HTTP_PROXY:-}
- HTTPS_PROXY=${HTTPS_PROXY:-}
```

那么 Docker Compose 在渲染时可能优先吃当前 shell 或 systemd 环境里的全局代理变量。  
结果就是：

1. `.env` 里明明写的是一个值。
2. 真正跑进容器里的却是另一个值。
3. 排查时非常容易误判。

更稳的写法是给该服务单独起名字：

```yaml
- HTTP_PROXY=${SUB2API_OUTBOUND_HTTP_PROXY:-}
- HTTPS_PROXY=${SUB2API_OUTBOUND_HTTPS_PROXY:-}
- NO_PROXY=${SUB2API_OUTBOUND_NO_PROXY:-localhost,127.0.0.1,::1,postgres,redis,host.docker.internal}
- http_proxy=${SUB2API_OUTBOUND_HTTP_PROXY:-}
- https_proxy=${SUB2API_OUTBOUND_HTTPS_PROXY:-}
- no_proxy=${SUB2API_OUTBOUND_NO_PROXY:-localhost,127.0.0.1,::1,postgres,redis,host.docker.internal}
```

这样容器自己的代理配置就不会被宿主机全局环境偷偷覆盖。

## 4. 一套通用排查方法

### ① 看 compose 渲染后的最终结果

```bash
docker compose -f docker-compose.local.yml config
```

重点看：

1. `dns` 是否真的出现在目标服务上。
2. `HTTP_PROXY` / `HTTPS_PROXY` 最终到底是什么。

### ② 看运行中容器实际吃到的配置

```bash
docker inspect <container_name>
```

重点看：

1. `HostConfig.Dns`
2. `HostConfig.DnsSearch`
3. `Config.Env`

### ③ 进入容器验证 DNS

```bash
docker compose exec -T sub2api sh -lc '
cat /etc/resolv.conf
getent hosts api.qinzhiai.com || true
getent hosts chatgpt.com || true
'
```

如果修好后能看到公网 IP，说明 DNS 基本打通了。

### ④ 显式关闭代理做直连测试

BusyBox 的 `wget` 可以用 `-Y off` 关代理：

```bash
docker compose exec -T sub2api sh -lc '
wget -Y off -S --spider --timeout=8 https://api.qinzhiai.com/v1
'
```

注意：

1. 返回 `404` 不代表失败，只说明“域名可达，路径不存在或不是 GET 入口”。
2. 返回 `403` 也不代表 DNS 有问题，往往说明已经成功连到上游。
3. 真正要怕的是 `bad address`、`server misbehaving`、`Resolving timed out`。

## 5. 这次 sub2api 的结论

这次 `sub2api` 最后验证通过，说明这套方法是有效的：

1. 给 `sub2api` 单独配 `223.5.5.5` 和 `119.29.29.29`
2. 清空 `dns_search`
3. 让代理变量改读 `SUB2API_OUTBOUND_*`
4. 重建容器

修完后：

1. 容器里能解析 `api.qinzhiai.com`
2. 容器里无代理直连能连上上游
3. `codex-kimi -> kimi-k2.6` 的真实请求恢复成功

## 6. 经验总结

在 WSL + Docker 环境里，如果只是“某几个容器访问公网域名失败”，优先顺序我现在更推荐：

1. 先看是不是 Docker 默认 DNS 被宿主机当前网络环境污染了。
2. 再看 compose 有没有被宿主机全局 `HTTP_PROXY` / `HTTPS_PROXY` 偷偷覆盖。
3. 优先做 **per-container DNS 修复**，不要一上来改整个系统 DNS。
4. 对容易受环境影响的服务，尽量使用自己专用的环境变量名，例如 `SUB2API_*`、`TOOLHUB_*`。

这种修法比“全局改 WSL DNS”更稳，也更不容易误伤别的服务。
