# Dockerfile for the frontend service

FROM node:24 AS dev
WORKDIR /frontend
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0"]
