import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import time
import os
import re

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://sldict.korean.go.kr/front/sign/signContentsView.do"


def sanitize_filename(filename: str) -> str:
    
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()


def extract_origin_no_from_url(url: str) -> int | None:
    
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'origin_no' in params:
            return int(params['origin_no'][0])
    except Exception as e:
        print(f"[WARN] URL에서 origin_no 추출 실패: {e}")
    return None


def get_rendered_soup(origin_no: int, retry_count: int = 3, debug: bool = True) -> BeautifulSoup | None:
    

    for attempt in range(retry_count):
        options = Options()

       
        if not debug:
            options.add_argument("--headless=new")

        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--window-size=1920,1080")

        # 방화벽/SSL 우회 옵션
        options.add_argument("--ignore-certificate-errors") 
        options.add_argument("--ignore-ssl-errors")
        options.add_argument("--allow-insecure-localhost")

       
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0')

       
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        options.page_load_strategy = 'normal'

        driver = None

        try:
            print(f"[DEBUG] ChromeDriver 초기화 중...")
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(40)  # 타임아웃 증가

            url = f"{BASE_URL}?origin_no={origin_no}"

            if attempt > 0:
                print(f"[RETRY] 재시도 {attempt}/{retry_count-1}...")

            print(f"[DEBUG] URL 접근 시도: {url}")
            driver.get(url)

            print(f"[DEBUG] 페이지 로드 완료, 현재 URL: {driver.current_url}")
            print(f"[DEBUG] 페이지 제목: {driver.title}")

           
            print(f"[DEBUG] Ajax 비디오 로딩 대기 중...")
            time.sleep(3) 

           
            print(f"[DEBUG] 비디오 컨테이너 확인 중...")
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "videoArea"))
            )
            print(f"[DEBUG] videoArea 발견")

           
            print(f"[DEBUG] 비디오 태그 탐색 중...")
            video_element = None

            
            try:
                video_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "#videoArea video"))
                )
                print(f"[DEBUG] 비디오 태그 발견: {video_element.tag_name}")
            except:
                print(f"[DEBUG] #videoArea video 찾기 실패, 다른 선택자 시도...")

            # 시도 2: 전체 video 태그 탐색
            if not video_element:
                try:
                    video_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "video"))
                    )
                    print(f"[DEBUG] 비디오 태그 발견 (전체 탐색): {video_element.tag_name}")
                except:
                    print(f"[DEBUG] video 태그 전혀 발견되지 않음")

            # source 태그 확인
            if video_element:
                print(f"[DEBUG] source 태그 확인 중...")
                try:
                    source_element = video_element.find_element(By.TAG_NAME, "source")
                    src_url = source_element.get_attribute("src")
                    print(f"[DEBUG] source 태그 발견, src: {src_url}")
                except:
                    print(f"[DEBUG] source 태그 찾기 실패")

            # 추가 안정화 대기
            time.sleep(1)

            html = driver.page_source

            if debug:
                # HTML 일부 저장 (디버깅용)
                debug_file = f"debug_origin_{origin_no}.html"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"[DEBUG] HTML 저장됨: {debug_file}")

            return BeautifulSoup(html, "html.parser")

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            print(f"[DEBUG] 에러 타입: {error_type}")
            print(f"[DEBUG] 에러 상세: {error_msg}")

         
            if driver and debug:
                try:
                    screenshot_file = f"debug_error_origin_{origin_no}_attempt_{attempt}.png"
                    driver.save_screenshot(screenshot_file)
                    print(f"[DEBUG] 스크린샷 저장: {screenshot_file}")
                except:
                    pass

            if attempt < retry_count - 1:
                print(f"[WARN] 시도 {attempt + 1} 실패: {error_msg}, 재시도 중...")
                time.sleep(3) 
                continue
            else:
                print(f"[ERROR] Selenium 페이지 로드 실패 (origin_no={origin_no})")
                print(f"[ERROR] 최종 에러: {error_type} - {error_msg}")
                return None
        finally:
            if driver:
                driver.quit()

    return None


def get_word_from_soup(soup: BeautifulSoup) -> str | None:
    """Soup에서 단어 텍스트 추출"""
    try:
        dl_tag = soup.find('dl', class_='content_view_dis')
        if dl_tag:
            dd_tag = dl_tag.find('dd')
            if dd_tag:
                return dd_tag.get_text(strip=True)
    except Exception as e:
        print(f"[WARN] 단어 추출 실패: {e}")
    return None


def get_video_url_from_soup(soup: BeautifulSoup, page_url: str) -> str | None:
    
    try:
       
        video_tag = soup.find('video')
        if not video_tag:
            print("[DEBUG] <video> 태그를 찾지 못함")
            return None

        print(f"[DEBUG] video 태그 발견")

       
        source_tag = video_tag.find(
            'source',
            attrs={'src': lambda s: s and (s.endswith('.mp4') or s.endswith('.webm'))}
        )

        if source_tag and source_tag.get('src'):
            src_url = source_tag.get('src')
            print(f"[DEBUG] source src 속성: {src_url}")
            full_url = urljoin(page_url, src_url)
            print(f"[DEBUG] 완전한 URL: {full_url}")
            return full_url

        
        if video_tag.get('src'):
            src_url = video_tag.get('src')
            print(f"[DEBUG] video src 속성: {src_url}")
            return urljoin(page_url, src_url)

        print("[DEBUG] source/video src를 찾지 못함")

    except Exception as e:
        print(f"[WARN] 비디오 URL 추출 실패: {e}")
    return None


def download_sign_video(origin_no: int, save_dir: str, debug: bool = True) -> dict:
    
    page_url = f"{BASE_URL}?origin_no={origin_no}"
    
    try:
        
        print(f"[DEBUG] origin_no={origin_no} 페이지 렌더링 시작...")
        soup = get_rendered_soup(origin_no, debug=debug)
        if soup is None:
            return {"status": "failed", "error": "Selenium page load failed", "origin_no": origin_no}

        
        word = get_word_from_soup(soup)
        print(f"[DEBUG] 추출된 단어: {word}")

        video_url = get_video_url_from_soup(soup, page_url)
        print(f"[DEBUG] 추출된 비디오 URL: {video_url}")

        if not video_url:
            return {"status": "failed", "error": "video src not found", "origin_no": origin_no}

       
        if word:
            filename = sanitize_filename(word) + ".mp4"
        else:
            filename = f"sign_video_{origin_no}.mp4"

        save_path = os.path.join(save_dir, filename)
        print(f"[DEBUG] 저장 경로: {save_path}")

        
        print(f"[DEBUG] 비디오 다운로드 시작...")
        with requests.get(video_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            print(f"[DEBUG] HTTP 응답 상태: {resp.status_code}")
            print(f"[DEBUG] Content-Type: {resp.headers.get('Content-Type')}")

            os.makedirs(save_dir, exist_ok=True)
            total_size = 0
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)

            print(f"[DEBUG] 다운로드 완료, 파일 크기: {total_size} bytes ({total_size/1024:.2f} KB)")

        return {
            "status": "success",
            "path": save_path,
            "word": word,
            "origin_no": origin_no
        }

    except Exception as e:
        print(f"[DEBUG] 다운로드 중 에러 발생: {type(e).__name__} - {str(e)}")
        return {"status": "failed", "error": repr(e), "origin_no": origin_no}


def batch_download_videos_from_urls(
    url_list: list[str],
    output_dir: str = "video_downloads",
    txt_file: str = "video_downloads/downloaded_list.txt",
    delay_seconds: float = 3.0,  
    max_retries: int = 2,  
    debug: bool = True  
) -> dict:
   

    downloaded_files = []
    failed_files = []
    os.makedirs(output_dir, exist_ok=True)

    
    origin_nos = []
    for url in url_list:
        clean_url = url.strip() 
        origin_no = extract_origin_no_from_url(clean_url)
        if origin_no:
            origin_nos.append(origin_no)
        else:
            print(f"[WARN] URL에서 origin_no를 찾을 수 없음: {clean_url}")

    total_count = len(origin_nos)
    
    print(f"\n{'=' * 60}")
    print(f"수어 영상 다운로드 시작: {total_count}개")
    print(f"저장 경로: {output_dir}")
    print(f"다운로드 간격: {delay_seconds}초")
    print(f"{'=' * 60}\n")

    for idx, origin_no in enumerate(origin_nos, 1):
        print(f"\n[{idx}/{total_count}] origin_no={origin_no} 다운로드 중...")

        
        result = None
        for retry in range(max_retries + 1):
            if retry > 0:
                print(f"[RETRY] 다운로드 재시도 {retry}/{max_retries}...")
                time.sleep(3)  

            result = download_sign_video(origin_no, output_dir, debug=debug)

            if result["status"] == "success":
                downloaded_files.append(result["path"])
                print(f"[SUCCESS] {result['word']} -> {result['path']}")
                break
            elif retry < max_retries:
                print(f"[WARN] 다운로드 실패, 재시도 중... (에러: {result['error']})")
            else:
                failed_files.append({"origin_no": origin_no, "error": result["error"]})
                print(f"[FAILED] origin_no={origin_no}, error={result['error']}")

        
        if idx < total_count:  
            print(f"[WAIT] {delay_seconds}초 대기 중...")
            time.sleep(delay_seconds)

    
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    with open(txt_file, "w", encoding="utf-8") as f:
        for path in downloaded_files:
            f.write(path + "\n")

    print(f"\n{'=' * 60}")
    print(f"다운로드 완료! 성공: {len(downloaded_files)}개 | 실패: {len(failed_files)}개")
    print(f"리스트 파일: {txt_file}")
    if failed_files:
        print("\n실패 목록:")
        for fail in failed_files:
            print(f"  - origin_no={fail['origin_no']}: {fail['error']}")
    print(f"{'=' * 60}\n")

    return {
        "success": len(downloaded_files),
        "failed": len(failed_files),
        "list_file": txt_file,
        "failed_list": failed_files
    }


if __name__ == "__main__":
    
   
    urls = [
        "https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=11762&top_category=CTE&category=&searchKeyword=%E3%85%A2&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0",
        "https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=9816&top_category=CTE&category=&searchKeyword=%E3%85%91&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0"
    ]
    
    
    result = batch_download_videos_from_urls(
        url_list=urls,
        output_dir="video_downloads",
        txt_file="video_downloads/downloaded_list.txt",
        delay_seconds=5.0,  # 지연시간
        max_retries=2,  # 재시도 횟수
        debug=False  # 디버그 모드: False = headless, True = 브라우저 표시
    )
    
    print("\n최종 결과:", result)