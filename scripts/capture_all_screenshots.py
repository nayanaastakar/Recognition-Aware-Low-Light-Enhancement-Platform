import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUT_DIR = os.path.abspath(r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement\docs\screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

options = Options()
options.add_argument("--headless=new")
options.add_argument("--window-size=1920,1080")
options.add_argument("--force-device-scale-factor=1")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=options)
driver.set_window_size(1920, 1080)

def snap_view(filename, target_element=None, delay=1.0):
    time.sleep(delay)
    if target_element is not None:
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'instant', block: 'center'});", target_element)
        time.sleep(0.8)
    filepath = os.path.join(OUT_DIR, filename)
    driver.save_screenshot(filepath)
    print(f"Captured: {filename} ({os.path.getsize(filepath)} bytes)")

def find_by_text(tag, text_sub):
    for el in driver.find_elements(By.TAG_NAME, tag):
        try:
            if text_sub in el.text:
                return el
        except Exception:
            pass
    return None

try:
    print("Navigating to dashboard...")
    driver.get("http://localhost:8501")
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"))
    )
    print("Page loaded. Waiting 10s for initial inference passes...")
    time.sleep(10)

    # 1. Dashboard Overview
    snap_view("01_dashboard_overview.png")

    # 2. Pipeline Architecture
    exp = find_by_text("summary", "Pipeline Architecture") or find_by_text("div", "Pipeline Architecture")
    if exp:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", exp)
        time.sleep(0.5)
        exp.click()
        time.sleep(1.0)
        snap_view("02_pipeline_architecture.png", target_element=exp)
        # click to close
        exp.click()
        time.sleep(0.5)

    # 3. Four-Way Side-by-Side Visual Comparison
    comp_header = find_by_text("div", "Four-Way Side-by-Side Visual Comparison")
    if comp_header:
        snap_view("03_four_way_comparison.png", target_element=comp_header)

    # 4. Model Performance Analytics - Tab 1: Detection & Latency Dynamics
    analytics_header = find_by_text("div", "Model Performance Analytics")
    if analytics_header:
        snap_view("04_analytics_detection_latency.png", target_element=analytics_header)

    # 5. Model Performance Analytics - Tab 2: Luminance & Contrast Spread
    tabs = driver.find_elements(By.CSS_SELECTOR, "button[data-baseweb='tab']")
    for t in tabs:
        if "Luminance" in t.text:
            t.click()
            time.sleep(1.0)
            snap_view("05_analytics_luminance_contrast.png", target_element=analytics_header)
            break

    # 6. Model Performance Analytics - Tab 3: Benchmark Transparency Matrix
    for t in tabs:
        if "Transparency" in t.text:
            t.click()
            time.sleep(1.0)
            snap_view("06_analytics_transparency_matrix.png", target_element=analytics_header)
            break

    # 7. Benchmark Verification Module Overview
    bench_header = find_by_text("div", "Ground-Truth Benchmark Verification Module")
    if bench_header:
        snap_view("07_benchmark_verification_overview.png", target_element=bench_header)

    # 8. Run Benchmark on yellow_bicycle_bench
    run_btn = find_by_text("button", "Run Benchmark with Verified Ground Truth")
    if run_btn:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", run_btn)
        time.sleep(0.8)
        run_btn.click()
        print("Executing benchmark on yellow_bicycle_bench...")
        time.sleep(6)
        
        bench_res_hdr = find_by_text("h4", "Ground-Truth Benchmark Results")
        if bench_res_hdr:
            snap_view("08_exdark_benchmark_results_kpis.png", target_element=bench_res_hdr)

        # Open match details expander
        iou_exp = find_by_text("summary", "Bounding Box IoU Matches") or find_by_text("div", "Bounding Box IoU Matches")
        if iou_exp:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", iou_exp)
            time.sleep(0.5)
            iou_exp.click()
            time.sleep(1.0)
            snap_view("09_bounding_box_iou_matches.png", target_element=iou_exp)

    # 9. Detected Objects Detailed Inspection
    det_insp = find_by_text("div", "Detected Objects Detailed Inspection")
    if det_insp:
        snap_view("10_detected_objects_inspection.png", target_element=det_insp)

    # 10. Switch to 2015_00026 Scene
    print("Switching scene to 2015_00026.jpg...")
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1.0)
    
    # Click sidebar selectbox
    sboxes = driver.find_elements(By.CSS_SELECTOR, "[data-testid='stSelectbox']")
    if sboxes:
        sboxes[0].click()
        time.sleep(1.0)
        options = driver.find_elements(By.CSS_SELECTOR, "li[role='option']")
        for o in options:
            if "2015_00026" in o.text:
                o.click()
                print("Selected 2015_00026.jpg. Waiting 10s for full inference...")
                time.sleep(10)
                break

        # 11. 2015_00026 Visual Comparison
        comp_hdr_2 = find_by_text("div", "Four-Way Side-by-Side Visual Comparison")
        if comp_hdr_2:
            snap_view("11_scene_00026_four_way_comparison.png", target_element=comp_hdr_2)

        # 12. Run benchmark on 2015_00026
        run_btn_2 = find_by_text("button", "Run Benchmark with Verified Ground Truth")
        if run_btn_2:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", run_btn_2)
            time.sleep(0.8)
            run_btn_2.click()
            print("Executing benchmark on 2015_00026.jpg...")
            time.sleep(6)
            
            bench_res_hdr_2 = find_by_text("h4", "Ground-Truth Benchmark Results")
            if bench_res_hdr_2:
                snap_view("12_scene_00026_benchmark_results.png", target_element=bench_res_hdr_2)

    print("All captures completed successfully!")

finally:
    driver.quit()
