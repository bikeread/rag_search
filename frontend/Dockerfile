# 使用Node.js 20作为基础镜像  
FROM node:20-alpine

# 设置工作目录
WORKDIR /app

# 复制package.json
COPY package.json ./

# 安装依赖（包括devDependencies）
RUN npm install

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 3000

# 构建生产版本
RUN npm run build

# 安装serve用于生产环境
RUN npm install -g serve

# 启动应用 - 使用serve运行构建后的文件
CMD ["serve", "-s", "dist", "-l", "3000"]