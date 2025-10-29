import re
import ipaddress
import numpy as np
import pandas as pd
from urllib.parse import urlparse, ParseResult
from math import log2
from sklearn.base import BaseEstimator, TransformerMixin

def normalize_url(u: str) -> str:
    '''
    Функция для нормализации URL: 
    
    1. Оставляем только строки
    2. Убираем невидимые символы (BOM, soft hyphen, zero-width)
    3. Добавляем схему (http://), если ее нет
    4. Убираем пробелы

    Возвращаем нормализованный URL
    '''
    if not isinstance(u, str):
        return ""
    u = re.sub(r'[\u200B-\u200D\uFEFF\u00AD]', '', u)
    u = u.encode("utf-8", "ignore").decode("utf-8", "ignore")
    u = u.strip()
    if not u:
        return u
    if not re.match(r'^\w+://', u):
        u = "http://" + u

    return u


def shannon_entropy(s: str) -> float:
    '''
    Функция для вычисления энтропии строки (чем выше, тем более подозрительный URL):

    1. Если строка пустая, то аозвращаем 0.0
    2. Считаем вероятность появления уникальных символов
    3. Вычисляем энтропию по формуле Шеннона

    Возвращаем числовое значение для каждой строки
    '''
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]

    return -sum(p * log2(p) for p in probs)


def parse_url(url: str) -> ParseResult:
    '''
    Функция для парсинга URL и IP:

    1. Нормализуем URL
    2. Парсим URL на компоненты
    3. Если хост — это IP-адрес (IPv4 или IPv6), возвращаем результат с IP-адресом
    4. Поддержка IPv6 с квадратными скобками

    Возвращаем компоненты URL
    '''
    try:
        url = normalize_url(url)
        host = url.split("://", 1)[1].split("/")[0]
        try:
            ipaddress.ip_address(host.strip("[]"))
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            return ParseResult(scheme=url.split("://", 1)[0],
                               netloc=host,
                               path="", params="", query="", fragment="")
        except ValueError:
            return urlparse(url)
    except Exception:
        return ParseResult(scheme="", netloc="", path="", params="", query="", fragment="")


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    '''
    Класс для извлечения признаков из URL-ов для машинного обучения
    '''
    def __init__(self):
        '''
        1. Инициализация класса
        2. Задаем подозрительные ключевые слова и названия извлекаемых признаков
        '''
        self.suspicious_keywords = ["login","secure","verify","update","bank",
                                    "account","free","bonus","click","win"]
        
        self.feature_names_ = ["url_length","num_digits","num_special_chars",
                               "num_subdomains","has_ip","num_slashes",
                               "domain_length","has_suspicious_keyword",
                               "has_suspicious_tld","domain_entropy",
                               "num_query_params","query_length",
                               "has_https","has_at_symbol"]

    def fit(self, X, y=None):
        '''
        Метод для обучения. Не делает ничего, возвращает сам себя
        '''
        return self

    def transform(self, X):
        '''
        Функция для извлечения признаков из X (колонка с URL-ами):

        1. Парсим URL (при помощи функции parse_url)
        2. Извлекаем признаки

        Возвращаем DataFrame с этими признаками
        '''
        urls = X if isinstance(X,(list,np.ndarray,pd.Series)) else X['url']
        features = []
        for url in urls:
            parsed = parse_url(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query = parsed.query

                # Длина URL
            f = {"url_length": len(url),
                 # Количество цифр
                 "num_digits": sum(c.isdigit() for c in url),
                 # Количество специальных символов (@?=&%)
                 "num_special_chars": len(re.findall(r'[@?=&%]', url)),
                 # Количество поддоменов
                 "num_subdomains": max(domain.count('.') - 1, 0),
                 # Количество слэшей
                 "num_slashes": url.count('/'),
                 # Длина домена
                 "domain_length": len(domain),
                 # Наличие подозрительных ключевых слов
                 "has_suspicious_keyword": int(any(kw in (domain+path) for kw in self.suspicious_keywords)),
                 # Наличие подозрительного TLD
                 "has_suspicious_tld": int(domain.endswith((".xyz",".top",".gq",".tk",".ml",".icu",".cn"))),
                 # Энтропия домена (при помощи функции shannon_entropy)
                 "domain_entropy": shannon_entropy(domain),
                 # Количество параметров запроса
                 "num_query_params": len(re.findall(r'=', query)),
                 # Длина строки запроса
                 "query_length": len(query),
                 # Наличие HTTPS
                 "has_https": 1 if parsed.scheme == "https" else 0,
                 # Наличие @
                 "has_at_symbol": 1 if "@" in url else 0}
            # Есть ли IP‑адрес
            try:
                ipaddress.ip_address(domain.strip("[]"))
                f["has_ip"] = 1
            except ValueError:
                f["has_ip"] = 0

            features.append(f)

        return pd.DataFrame(features, columns=self.feature_names_)

    def get_feature_names_out(self, input_features=None):
        '''
        Возвращаем список названий признаков
        '''
        return np.array(self.feature_names_)


class DomainTokenizer(BaseEstimator, TransformerMixin):
    '''
    Класс для извлечения доменного имени в виде токенов для машинного обучения
    '''
    def __init__(self):
        '''
        1. Инициализация класса
        2. Задаем название извлекаемого признака
        '''
        self.feature_names_ = ["domain_tokens"]

    def fit(self, X, y=None):
        '''
        Метод для обучения. Не делает ничего, возвращает сам себя
        '''
        return self

    def transform(self, X):
        '''
        Функция для трансформации URL-ы в токены

        1. Ивлекаем доменную часть
        2. Меняем точки и дефисы в домене на пробелы

        Возвращаем массив токенов
        '''
        urls = X if isinstance(X,(list,np.ndarray,pd.Series)) else X['url']
        tokens = []
        for url in urls:
            url = normalize_url(url)
            domain = urlparse(url).netloc.lower()
            tokens.append(re.sub(r'[\.-]', ' ', domain))

        return np.array(tokens)

    def get_feature_names_out(self, input_features=None):
        '''
        Возвращаем список названий признаков (1 признак - 'domain_tokens')
        '''
        return np.array(self.feature_names_)