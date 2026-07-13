/// <reference types="vite/client" />

declare global {
  interface Window {
    renderMathInElement?: (
      element: HTMLElement,
      options?: {
        delimiters?: Array<{ left: string; right: string; display: boolean }>;
        throwOnError?: boolean;
      },
    ) => void;
  }
}

export {};
