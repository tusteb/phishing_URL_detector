import re
import requests
import logging
import ipaddress
from urllib.parse import urlparse

logger = logging.getLogger("uvicorn.error")

def load_tlds():
    '''
    Функция для загрузки списка TLD (доменных зон верхнего уровня) из IANA
    При ошибке возвращаем минимальный набор TLD
    '''
    url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        tlds = {line.strip().lower() for line in resp.text.splitlines() if not line.startswith("#")}
        logger.info(f"Загружено {len(tlds)} TLD из IANA")
        return tlds
    except Exception as e:
        logger.warning(f"Не удалось загрузить список TLD из IANA: {e}")
        return {"com", "org", "net", "ru", "xyz", "tk", "io", "dev", "info"}

VALID_TLDS = load_tlds()

def is_valid_url_or_ip(u: str) -> bool:
    '''
    Функция для проверки валидности URL/IP:

    Если не IP, то добавляет схему, парсит URL и проверяет:
            что схема — http или https,
            что есть netloc (домен),
            что в netloc нет недопустимых символов и пробелов
    '''
    if not u or not u.strip():
        return False
    u = u.strip()

    # Проверка IP (IPv4/IPv6)
    try:
        ipaddress.ip_address(u.strip("[]"))
        return True
    except ValueError:
        pass

    # Проверка домена/URL
    try:
        if not re.match(r'^\w+://', u):
            u = "http://" + u
        result = urlparse(u)

        if result.scheme not in ("http", "https"):
            return False
        if not result.netloc:
            return False

        # Убираем пробелы и недопустимые символы
        if " " in result.netloc:
            return False
        if not re.match(r'^[A-Za-z0-9\.\-\[\]:]+$', result.netloc):
            return False

        return True
    except Exception:
        return False


def is_plausible_url(url: str) -> bool:
    '''
    Функция для проверки URL на правдоподобие (что это не случайная строка):

    1. Парсим URL и извлекаем домен
    2. Проверяем наличие точки и что TLD есть в списке VALID_TLDS
    3. Проверяем, что в домене есть хотя бы одна латинская буква
    '''
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain or "." not in domain:
            return False

        tld = domain.split(".")[-1]
        if tld not in VALID_TLDS:
            return False
        return bool(re.search(r"[a-zA-Z]", domain))
    except Exception:
        return False