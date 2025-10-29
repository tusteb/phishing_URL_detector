import streamlit as st
import requests
import pandas as pd
from utils import is_valid_url_or_ip
import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Настройки страницы
st.set_page_config(page_title="Phishing URL Detector")

st.markdown("# 🔍 Проверка URL на фишинг")
st.markdown("Введите URL или IP, чтобы получить предсказание модели, уровень уверенности и объяснение.")

# Ввод URL/IP
url = st.text_input("Введите URL или IP:")

if st.button("**Проверить**", use_container_width=True):
    if not url:
        st.warning("⚠️ Пожалуйста, введите URL или IP.")
    elif not is_valid_url_or_ip(url):
        st.error("❌ Введённый текст не похож на корректный URL или IP‑адрес.")
    else:
        # Запрос к /predict
        try:
            response = requests.get(f"{API_URL}/predict", params={"url": url})
            if response.status_code == 200:
                result = response.json()
            else:
                try:
                    error_detail = response.json().get("detail", "Неизвестная ошибка")
                except Exception:
                    error_detail = response.text
                st.error(f"❌ Ошибка API ({response.status_code}): {error_detail}")
                st.stop()
        except Exception as e:
            st.error(f"⚠️ Ошибка при обращении к API: {e}")
            st.stop()

        # Результат
        st.markdown("## Результат модели")
        if result["prediction"] == 1:
            st.error("🟠 Фишинговый URL")
        else:
            st.success("🟢 Безопасный URL")

        # Метрика
        st.metric("Уверенность модели", f"{result['probability']*100:.1f}%")

        # Кнопка "скачать результат"
        report = "\n".join([f"Результат проверки URL: {result['url']}",
                            f"Предсказание модели: {'Фишинг' if result['prediction'] == 1 else 'Безопасно'}",
                            f"Уверенность: {result['probability']*100:.1f}%",
                            f"Доверенный домен: {'Да' if result.get('trusted') else 'Нет'}"])
        st.download_button(label="📥 Скачать результат",
                           data=report,
                           file_name="report.txt",
                           mime="text/plain",
                           use_container_width=True)

        # Запрос к /explain
        try:
            explain_resp = requests.get(f"{API_URL}/explain", params={"url": url})
            if explain_resp.status_code == 200:
                explain_data = explain_resp.json()
                explanations = explain_data["explanations"]

                st.markdown("## Эвристический анализ")
                if explain_data.get("trusted"):
                    st.markdown("✅ URL признан безопасным, так как домен входит в белый список.")

                # Таблица эвристик с расшифровкой
                df = pd.DataFrame(explanations)
                df["emoji"] = df["color"].map({"green": "🟢",
                                               "yellow": "🟡",
                                               "red": "🔴"})
                df["value"] = df["value"].apply(lambda v: ", ".join(v) if isinstance(v, list) and len(v) > 0 else (0 if isinstance(v, list) else v))

                feature_descriptions = {"url_length": "Длина URL",
                                        "num_dots": "Количество точек в адресе",
                                        "num_special_chars": "Количество специальных символов (@, -, _, =, %, ?)",
                                        "entropy": "Случайность доменного имени (энтропия)",
                                        "is_ip_address": "Используется ли IP вместо домена",
                                        "num_subdomains": "Количество поддоменов",
                                        "suspicious_keywords": "Подозрительные слова в адресе",
                                        "suspicious_path_keywords": "Подозрительные слова в пути",
                                        "tld": "Доменная зона (TLD)",
                                        "is_suspicious_tld": "Подозрительная доменная зона",
                                        "is_trusted_domain": "Доверенный домен"}

                df_display = df[["emoji", "feature", "value"]].rename(columns={"emoji": "",
                                                                               "feature": "Признак",
                                                                               "value": "Значение"})
                df_display["Признак"] = df_display["Признак"].map(feature_descriptions)
                df_display["Значение"] = df_display["Значение"].astype(str)
                st.table(df_display)
            else:
                try:
                    error_detail = explain_resp.json().get("detail", "Неизвестная ошибка")
                except Exception:
                    error_detail = explain_resp.text
                st.warning(f"⚠️ Ошибка API при объяснении ({explain_resp.status_code}): {error_detail}")
        except Exception as e:
            st.warning(f"⚠️ Не удалось получить объяснение: {e}")