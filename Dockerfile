FROM public.ecr.aws/lambda/nodejs:20

COPY package.json package-lock.json* ./
RUN npm install

COPY src/ ./src/

CMD ["src/index.handler"]
