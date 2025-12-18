import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestAnalytics(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)
        
        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))
        self.driver.get(self.base_url + "/analytics")
        wait.until(EC.presence_of_element_located((By.ID, "totalPredictions")))

    def teardown_method(self):
        self.tear_down()

    def test_analytics_summary_cards(self):
        wait = WebDriverWait(self.driver, 10)
        total = wait.until(EC.visibility_of_element_located((By.ID, "totalPredictions")))
        avg = self.driver.find_element(By.ID, "averageGrade")
        excellent = self.driver.find_element(By.ID, "excellentCount")
        risk = self.driver.find_element(By.ID, "riskCount")
        
        assert total.is_displayed()
        assert avg.is_displayed()
        assert excellent.is_displayed()
        assert risk.is_displayed()

    def test_charts_initialization(self):
        wait = WebDriverWait(self.driver, 15)
        charts = [
            "gradeDistributionChart",
            "performanceLevelsChart",
            "modelComparisonChart",
            "predictionTrendsChart"
        ]
        
        for chart_id in charts:
            chart_element = wait.until(EC.presence_of_element_located((By.ID, chart_id)))
            assert chart_element.is_displayed()
            canvas_exists = self.driver.find_elements(By.CSS_SELECTOR, f"#{chart_id} .main-svg")
            assert len(canvas_exists) > 0

    def test_theme_persistence_on_charts(self):
        theme_btn = self.driver.find_element(By.CLASS_NAME, "theme-switcher")
        theme_btn.click()
        
        html_tag = self.driver.find_element(By.TAG_NAME, "html")
        current_theme = html_tag.get_attribute("data-bs-theme")
        
        chart_title = self.driver.find_element(By.CSS_SELECTOR, ".gtitle")
        assert chart_title.is_displayed()

    def test_navigation_back_to_dashboard(self):
        dashboard_link = self.driver.find_element(By.XPATH, "//a[contains(@href, '/dashboard')]")
        self.driver.execute_script("arguments[0].click();", dashboard_link)
        
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.url_contains("/dashboard"))
        assert "/dashboard" in self.driver.current_url

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])