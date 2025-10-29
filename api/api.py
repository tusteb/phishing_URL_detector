from fastapi import FastAPI, HTTPException, Query
import joblib
import pandas as pd
from model.transformers import normalize_url, shannon_entropy
from ui.utils import is_valid_url_or_ip, is_plausible_url
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from urllib.parse import urlparse
import json
import os
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Phishing URL Detector API")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

# Конфиг
BASE_DIR = Path(__file__).resolve().parent
THRESHOLD = float(os.getenv("PHISHING_THRESHOLD", 0.7)) # Порог для классификации (0.7)

CONFIG_PATH = BASE_DIR / "trusted_domains.json" # JSON с доверенными доменами 
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        TRUSTED_DOMAINS = json.load(f)["trusted_domains"]
except Exception:
    TRUSTED_DOMAINS = []

# Модель
MODEL_PATH = BASE_DIR.parent / "model" / "RF_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

# Постобработка
def postprocess_prediction(url: str, proba: float, threshold: float = THRESHOLD):
    '''
    Вспомогательная функция для обработки результатов модели:

    1. Парсим URL
    2. Извлекаем домен
    3. Убираем www.
    4. Проверяем, есть ли домен в trusted_domains.json

    Если домен доверенный - возвращаем статус "безопасный" (0), вероятность, и флаг trusted=True
    Если нет, то сравниваем вероятность с порогом и возвращаем предсказание (фишинг или безопасный), 
    вероятность и порог.
    '''
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    trusted = any(domain == td or domain.endswith("." + td) for td in TRUSTED_DOMAINS)
    if trusted:
        return 0, "safe", proba, threshold, True

    pred = 1 if proba >= threshold else 0
    class_name = "phishing" if pred == 1 else "safe"

    return pred, class_name, proba, threshold, False


# Эндпоинт /predict
@app.get("/predict")
def predict(url: str = Query(..., description="URL или IP для проверки")):
    '''
    Функция для получения предсказания:

    1. Получаем URL от пользователя
    2. Нормализуем его
    3. Проверяем, что URL не пустой, корректен и существует
    4. Создаем DataFrame с URL для модели
    5. Логируем тип и значение URL
    6. Предсказываем вероятность фишинга при помощи модели
    7. Обрабатываем результат при помощи postprocess_prediction

    Возвращаем JSON с информацией о предсказании: URL, класс, вероятность, порог, доверенность
    Если ошибка - логируем и возвращаем HTTP 500
    '''
    url = normalize_url(url)
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL не может быть пустым")
    if not is_valid_url_or_ip(url):
        raise HTTPException(status_code=400, detail="Некорректный формат URL или IP")
    if not is_plausible_url(url):
        raise HTTPException(status_code=400, detail="Некорректный или несуществующий домен")

    try:
        url = str(url)
        if not url:
            raise HTTPException(status_code=400, detail="URL не может быть пустым после нормализации")

        X = pd.DataFrame([[url]], columns=["url"])
        logger.info(f"Тип значения в X['url']: {type(X['url'].iloc[0])}, значение: {X['url'].iloc[0]!r}")

        proba = pipeline.predict_proba(X)[0, 1]
        pred, class_name, proba, threshold, trusted = postprocess_prediction(url, proba)

        logger.info(f"[PREDICT] URL={url}, proba={proba:.4f}, pred={pred} ({class_name}), trusted={trusted}")
        return {"url": url,
                "prediction": int(pred),
                "class_name": class_name,
                "probability": float(proba),
                "threshold": threshold,
                "trusted": trusted}

    except Exception as e:
        logger.exception("Ошибка при обработке URL")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке: {e}")


# Эндпоинт /explain
@app.get("/explain")
def explain(url: str = Query(..., description="URL или IP для объяснения")):
    '''
    Функция для получения объяснения результата:

    1. Получаем URL от пользователя
    2. Нормализуем его 
    3. Проверяем, есть ли домен в trusted_domains.json
    4. Вычисляем признаки:
            Длина URL
            Количество точек
            Количество спец. символов
            Энтропия
            Является ли URL IP-адресом
            Количество поддоменов
            Наличие подозрительных слов в URL
            Наличие подозрительных слов в пути
            Является ли tld подозрительным
            Является ли домен доверенным
    5. Задаем цвет для каждого признака по правилам
    6. Создаем heuristics, где каждому признаку сопоставляется значение и цвет

    Возвращаем JSON - описание признаков с цветовой маркировкой и флагом доверенности
    Логируем входные данные и результат 
    Если ошибка - логируем и возвращаем HTTP 500
    '''
    url = normalize_url(url)
    if not url.strip():
        raise HTTPException(status_code=400, detail="URL не может быть пустым")
    if not is_valid_url_or_ip(url):
        raise HTTPException(status_code=400, detail="Некорректный формат URL или IP")
    if not is_plausible_url(url):
        raise HTTPException(status_code=400, detail="Некорректный или несуществующий домен")

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        is_trusted = any(domain == td or domain.endswith("." + td) for td in TRUSTED_DOMAINS)
        path = parsed.path.lower()

        suspicious_words = ["login", "secure", "update", "verify", "account"]
        suspicious_path_words = ["claim", "bonus", "free", "gift", "win", "prize"]
        suspicious_tlds = ["tk", "xyz", "top", "gq", "cf"]
        url_length = len(url)
        num_dots = url.count(".")
        num_special = sum(url.count(c) for c in ["@", "-", "_", "=", "?", "%"])
        entropy_val = shannon_entropy(url)
        is_ip = domain.replace(".", "").isdigit()
        num_subdomains = len(domain.split(".")) - 2 if "." in domain else 0
        found_keywords = [w for w in suspicious_words if w in url.lower()]
        found_path_keywords = [w for w in suspicious_path_words if w in path]
        tld = domain.split(".")[-1] if "." in domain else ""
        is_bad_tld = tld in suspicious_tlds

        def colorize(feature, value):
            if feature == "url_length":
                if value < 30: return "green"
                elif value < 60: return "yellow"
                else: return "red"
            if feature == "num_dots":
                return "green" if value <= 2 else "red"
            if feature == "num_special_chars":
                return "green" if value <= 2 else "red"
            if feature == "entropy":
                return "green" if value < 4.0 else "red"
            if feature == "is_ip_address":
                return "red" if value else "green"
            if feature == "num_subdomains":
                return "green" if value <= 2 else "red"
            if feature == "suspicious_keywords":
                return "red" if value else "green"
            if feature == "suspicious_path_keywords":
                return "red" if value else "green"
            if feature == "is_suspicious_tld":
                return "red" if value else "green"
            return "yellow"

        heuristics = [{"feature": "url_length", "value": url_length, "color": colorize("url_length", url_length)},
                      {"feature": "num_dots", "value": num_dots, "color": colorize("num_dots", num_dots)},
                      {"feature": "num_special_chars", "value": num_special, "color": colorize("num_special_chars", num_special)},
                      {"feature": "entropy", "value": entropy_val, "color": colorize("entropy", entropy_val)},
                      {"feature": "is_ip_address", "value": is_ip, "color": colorize("is_ip_address", is_ip)},
                      {"feature": "num_subdomains", "value": num_subdomains, "color": colorize("num_subdomains", num_subdomains)},
                      {"feature": "suspicious_keywords", "value": found_keywords, "color": colorize("suspicious_keywords", found_keywords)},
                      {"feature": "suspicious_path_keywords", "value": found_path_keywords, "color": colorize("suspicious_path_keywords", found_path_keywords)},
                      {"feature": "tld", "value": tld, "color": "red" if is_bad_tld else "green"},
                      {"feature": "is_suspicious_tld", "value": is_bad_tld, "color": colorize("is_suspicious_tld", is_bad_tld)},
                      {"feature": "is_trusted_domain", "value": is_trusted, "color": "green" if is_trusted else "red"}]

        result = {"url": url,
                  "explanations": heuristics,
                  "detail": "Эвристические признаки URL с цветовой маркировкой",
                  "trusted": is_trusted}

        logger.info(f"[EXPLAIN] URL={url}, trusted={is_trusted}, heuristics_count={len(heuristics)}")

        return result

    except Exception as e:
        logger.exception("Ошибка при объяснении URL")
        raise HTTPException(status_code=500, detail=f"Ошибка при объяснении: {e}")