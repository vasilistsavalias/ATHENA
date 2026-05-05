"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { el } from "@/lib/i18n/el";
import { en } from "@/lib/i18n/en";
import { Dictionary, Locale } from "@/lib/i18n/types";

const I18N_STORAGE_KEY = "athena_locale";

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  dictionary: Dictionary;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("el");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const saved = window.localStorage.getItem(I18N_STORAGE_KEY);
    if (saved === "en" || saved === "el") {
      setLocaleState(saved);
      return;
    }
    const browser = navigator.language?.toLowerCase();
    if (browser?.startsWith("el")) {
      setLocaleState("el");
    }
  }, []);

  function setLocale(localeValue: Locale) {
    setLocaleState(localeValue);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(I18N_STORAGE_KEY, localeValue);
    }
  }

  const dictionary = useMemo(() => (locale === "el" ? el : en), [locale]);

  return <I18nContext.Provider value={{ locale, setLocale, dictionary }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within I18nProvider.");
  }
  return context;
}
