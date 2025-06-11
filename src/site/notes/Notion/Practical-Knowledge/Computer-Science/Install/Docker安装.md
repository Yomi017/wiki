---
{"dg-publish":true,"permalink":"/notion/practical-knowledge/computer-science/install/docker/"}
---

# Ubuntu 22.04 安装Docker

## 0. curl的安装

sudo apt update  
sudo apt install curl  

## 1. 安装方法

### ① 使用官方安装脚本自动安装

安装命令如下：

```Plain
 curl -fsSL https://test.docker.com -o test-docker.sh
 sudo sh test-docker.sh
```

（但是安装的可能是测试版而不是stable版）

### ② 手动安装

**卸载旧版本**

Docker 的旧版本被称为 docker，docker.io 或 docker-engine 。如果已安装，请卸载它们：

```Plain
sudo apt-get remove docker docker-engine docker.io containerd runc
```

当前称为 Docker Engine-Community 软件包 docker-ce 。

安装 Docker Engine-Community，以下介绍两种方式。

**使用 Docker 仓库进行安装**

在新主机上首次安装 Docker Engine-Community 之前，需要设置 Docker 仓库。之后，您可以从仓库安装和更新 Docker 。

**设置仓库**

更新 apt 包索引。

```Plain
sudo apt-get update
```

安装 apt 依赖包，用于通过HTTPS来获取仓库:

```Plain
sudo apt-get install \    apt-transport-https \    ca-certificates \    curl \    gnupg-agent \    software-properties-common
```

添加 Docker 的官方 GPG 密钥：

```Plain
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
```

使用以下指令设置稳定版仓库

```Plain
echo \  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/ \ $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \  sudo tee /etc/apt/sources.list.d/docker.list > /dev/nullsudo apt-get update
```

**安装 Docker Engine-Community**

更新 apt 包索引。

```Plain
sudo apt-get update
```

安装最新版本的 Docker Engine-Community 和 containerd ，或者转到下一步安装特定版本：

```Plain
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

要安装特定版本的 Docker Engine-Community，请在仓库中列出可用版本，然后选择一种安装。列出您的仓库中可用的版本：

```Plain
apt-cache madison docker-ce
apt-cache madison docker-ce docker-ce-cl
```

使用第二列中的版本字符串安装特定版本，例如 5:18.09.1~3-0~ubuntu-xenial。

```Plain
sudo apt-get install docker-ce=<版本号> docker-ce-cli=<版本号> containerd.io
```

## 2. 测试 Docker 是否安装成功

输入以下指令：

```Plain
sudo docker run hello-world
```

打印出以下信息则安装成功：

```JavaScript
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
1b930d010525: Pull complete                                                                                                                                  Digest: sha256:c3b4ada4687bbaa170745b3e4dd8ac3f194ca95b2d0518b417fb47e5879d9b5f
Status: Downloaded newer image for hello-world:latest


Hello from Docker!
This message shows that your installation appears to be working correctly.


To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.


To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash


Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/


For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## **3. 注意事项**

### ① 国内几乎不能访问Docker官网，sudo docker run hello-world可能不成功，因此需要更改镜像源

使用DaoCloud镜像源（至少2025.2.9前有效）：

首先，输入指令创建文件：

```JavaScript
sudo touch /etc/docker/daemon.json
```

在文件中写入

```JSON
{
  "registry-mirrors": ["https://docker.m.daocloud.io"]
}
```

若权限不足

首先，使用 `ls -l` 命令查看文件的当前权限设置：

```Shell
ls -l /etc/docker/daemon.conf
```

输出示例：

```Plain
-rw-r--r-- 1 root root 1234 Feb 9 02:26 /etc/docker/daemon.conf
```

  
使用  
`chown` 命令将文件的所有者和所属组更改为你的用户名：

```Plain
sudo chown <用户名>:<用户名> /etc/docker/example.conf
```

使用 `chmod` 命令为你的用户添加读写权限：

```Plain
sudo chmod u+rw /etc/docker/daemon.conf
```

再次使用 `ls -l` 命令确认权限已成功更改：

```Plain
ls -l /etc/docker/daemon.conf
```

输出示例：

```Plain
-rw-r--r-- 1 <用户名> <用户名> 1234 Feb 9 02:26 /etc/docker/daemon.conf
```

之后就可以修改文件了