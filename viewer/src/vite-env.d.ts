/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_HTTP_URL?: string;
  readonly VITE_API_WS_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
