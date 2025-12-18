import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from base_test import BaseTest
import time

class TestPrediction(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)
        
        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))

    def teardown_method(self):
        self.tear_down()

    def test_complete_prediction_flow(self):
        self.driver.find_element(By.ID, "hours_studied").clear()
        self.driver.find_element(By.ID, "hours_studied").send_keys("10")
        
        self.driver.find_element(By.ID, "attendance").clear()
        self.driver.find_element(By.ID, "attendance").send_keys("95")
        
        self.driver.find_element(By.ID, "previous_scores").clear()
        self.driver.find_element(By.ID, "previous_scores").send_keys("85")
        
        Select(self.driver.find_element(By.ID, "motivation_level")).select_by_value("High")
        
        self.driver.find_element(By.ID, "sleep_hours").clear()
        self.driver.find_element(By.ID, "sleep_hours").send_keys("8")
        
        Select(self.driver.find_element(By.ID, "internet_access")).select_by_value("Yes")
        
        self.driver.find_element(By.ID, "tutoring_sessions").clear()
        self.driver.find_element(By.ID, "tutoring_sessions").send_keys("2")
        
        self.driver.find_element(By.ID, "physical_activity").clear()
        self.driver.find_element(By.ID, "physical_activity").send_keys("5")

        predict_btn = self.driver.find_element(By.XPATH, "//button[contains(., 'Predict Performance')]")
        self.driver.execute_script("arguments[0].click();", predict_btn)

        wait = WebDriverWait(self.driver, 20)
        result_card = wait.until(EC.visibility_of_element_located((By.ID, "result")))
        assert result_card.is_displayed()
        
        score_badge = self.driver.find_element(By.ID, "predBadge")
        assert "/ 100" in score_badge.text

    def test_save_prediction_result(self):
        predict_btn = self.driver.find_element(By.XPATH, "//button[contains(., 'Predict Performance')]")
        self.driver.execute_script("arguments[0].click();", predict_btn)
        
        wait = WebDriverWait(self.driver, 20)
        save_btn = wait.until(EC.element_to_be_clickable((By.ID, "saveBtn")))
        self.driver.execute_script("arguments[0].click();", save_btn)
        
        wait.until(EC.text_to_be_present_in_element((By.ID, "saveBtn"), "Saved!"))
        assert "Saved" in self.driver.find_element(By.ID, "saveBtn").text

    def test_dynamic_impact_updates(self):
        hours_input = self.driver.find_element(By.ID, "hours_studied")
        hours_input.clear()
        hours_input.send_keys("20")
        
        time.sleep(1)
        study_impact = self.driver.find_element(By.ID, "studyImpact").text
        assert study_impact != "0%"

    def test_input_validation_clamping(self):
        hours_input = self.driver.find_element(By.ID, "hours_studied")
        hours_input.clear()
        hours_input.send_keys("150")
        self.driver.find_element(By.ID, "attendance").click()
        
        time.sleep(0.5)
        assert int(hours_input.get_attribute("value")) <= 100

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])