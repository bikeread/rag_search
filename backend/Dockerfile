FROM node:18-alpine

# 安装OpenSSL和其他必要的库
RUN apk update && apk add --no-cache openssl libc6-compat

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Generate Prisma client
RUN rm -rf node_modules/.prisma && npx prisma generate

# Expose port
EXPOSE 3000

# Start the application in development mode
CMD ["npm", "run", "dev"]