import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestModelVisuals(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)
        
        wait = WebDriverWait(self.driver, 20)
        wait.until(EC.url_to_be(self.base_url + "/"))
        
        self.driver.get(self.base_url + "/model_visuals")

    def teardown_method(self):
        self.tear_down()

    def test_page_and_charts_rendering(self):
        wait = WebDriverWait(self.driver, 30)
        
        header = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
        assert "Visualizations" in header.text
        
        charts = ["lossChart", "predActualChart", "featureImpChart", "residualsChart"]
        
        for chart_id in charts:
            wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, f"#{chart_id} .loading")))
            
            chart_svg = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, f"#{chart_id} .main-svg")))
            
            assert chart_svg.is_displayed()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])