FROM node:lts-alpine AS dev

WORKDIR /app

COPY src/ /app/src
COPY static/ /app/static
COPY .prettierrc /app/.prettierrc
COPY .prettierignore /app/.prettierignore
COPY eslint.config.js /app/eslint.config.js
COPY package.json /app/package.json
COPY package-lock.json /app/package-lock.json
COPY postcss.config.js /app/postcss.config.js
COPY svelte.config.js /app/svelte.config.js
COPY tailwind.config.js /app/tailwind.config.js
COPY tsconfig.json /app/tsconfig.json
COPY vite.config.ts /app/vite.config.ts

RUN npm install

EXPOSE 5173

ENTRYPOINT [ "npm", "run", "dev"]

FROM node:lts-alpine AS build

WORKDIR /app

# copy node modules from dev stage
COPY --from=dev /app/ /app/

RUN npm run build

FROM node:lts-alpine AS prod

WORKDIR /app

COPY --from=build /app/ /app/

WORKDIR /app/build

ENTRYPOINT [ "node", "index.js" ]
