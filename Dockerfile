# 使用Python 3.12 Alpine镜像作为基础镜像，体积更小
FROM python:3.12-alpine

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（Alpine需要的编译工具）
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    && rm -rf /var/cache/apk/*

# 复制依赖文件
COPY pyproject.toml ./

# 安装uv包管理器（更快的Python包管理器）
RUN pip install uv

# 使用uv安装依赖
RUN uv pip install --system --no-cache-dir -r pyproject.toml

# 复制应用代码
COPY main.py ./
COPY README.md ./

# 创建非root用户
RUN adduser -D -s /bin/sh appuser
USER appuser

# 暴露端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# 启动命令
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]