import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestInsights(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)
        
        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))
        
        self.driver.get(self.base_url + "/insights")
        wait.until(EC.presence_of_element_located((By.ID, "userName")))

    def teardown_method(self):
        self.tear_down()

    def test_insights_page_elements(self):
        wait = WebDriverWait(self.driver, 10)
        title = wait.until(EC.visibility_of_element_located((By.TAG_NAME, "h1")))
        assert "AI Insights" in title.text

    def test_ai_real_data_insights(self):
        wait = WebDriverWait(self.driver, 20)
        container = wait.until(EC.presence_of_element_located((By.ID, "realInsightsContainer")))
        
        wait.until(lambda d: "spinner-border" not in container.get_attribute("innerHTML"))
        
        cards = container.find_elements(By.CLASS_NAME, "insight-card")
        assert len(cards) > 0 or "No strong correlations" in container.text

    def test_model_intelligence_metrics(self):
        wait = WebDriverWait(self.driver, 20)
        
        accuracy_bar = wait.until(EC.visibility_of_element_located((By.ID, "accuracyProgress")))
        assert "%" in accuracy_bar.text
        
        r2 = self.driver.find_element(By.ID, "r2Value").text
        mae = self.driver.find_element(By.ID, "maeValue").text
        
        assert r2 != "0.00"
        assert mae != "0.00"

    def test_feature_importance_bars(self):
        wait = WebDriverWait(self.driver, 15)
        container = wait.until(EC.presence_of_element_located((By.ID, "featureImportanceContainer")))
        
        wait.until(lambda d: "spinner-border" not in container.get_attribute("innerHTML"))
        
        bars = container.find_elements(By.CLASS_NAME, "progress-bar")
        assert len(bars) > 0

    def test_improvement_plan_display(self):
        wait = WebDriverWait(self.driver, 20)
        container = wait.until(EC.presence_of_element_located((By.ID, "improvementPlanContainer")))
        
        wait.until(lambda d: "spinner-border" not in container.get_attribute("innerHTML"))
        assert "Target" in container.text
        assert "Current" in container.text

    def test_dataset_stats_loading(self):
        wait = WebDriverWait(self.driver, 15)
        container = wait.until(EC.presence_of_element_located((By.ID, "datasetStatsContainer")))
        
        wait.until(lambda d: "spinner-border" not in container.get_attribute("innerHTML"))
        assert "Total Students" in container.text

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])