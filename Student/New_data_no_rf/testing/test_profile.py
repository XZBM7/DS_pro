import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from base_test import BaseTest
import time

class TestProfile(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)
        
        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))
        self.driver.get(self.base_url + "/profile")
        wait.until(EC.presence_of_element_located((By.ID, "userFullName")))

    def teardown_method(self):
        self.tear_down()

    def test_profile_data_loading(self):
        name = self.driver.find_element(By.ID, "userFullName")
        assert name.text != "Loading..."
        assert "@" in self.driver.find_element(By.ID, "userEmail").text

    def test_update_full_name(self):
        full_name_input = self.driver.find_element(By.ID, "fullName")
        full_name_input.clear()
        full_name_input.send_keys("Ibrahim Amr")
        self.driver.find_element(By.ID, "profileForm").submit()
        
        wait = WebDriverWait(self.driver, 10)
        toast = wait.until(EC.visibility_of_element_located((By.ID, "toastMessage")))
        assert "Success" in self.driver.find_element(By.ID, "toastTitle").text

    def test_password_strength_weak(self):
        pwd_input = self.driver.find_element(By.ID, "newPassword")
        strength_bar = self.driver.find_element(By.ID, "passwordStrength")
        pwd_input.send_keys("123")
        assert "strength-weak" in strength_bar.get_attribute("class")

    def test_password_strength_very_strong(self):
        pwd_input = self.driver.find_element(By.ID, "newPassword")
        strength_bar = self.driver.find_element(By.ID, "passwordStrength")
        pwd_input.send_keys("Ibrahim@2025_Secure")
        assert "strength-very-strong" in strength_bar.get_attribute("class")

    def test_password_mismatch_ui(self):
        self.driver.find_element(By.ID, "newPassword").send_keys("NewPass123")
        confirm_input = self.driver.find_element(By.ID, "confirmPassword")
        confirm_input.send_keys("WrongPass")
        match_text = self.driver.find_element(By.ID, "passwordMatch")
        assert "do not match" in match_text.text
        assert "text-danger" in match_text.get_attribute("class")

    def test_password_match_success_ui(self):
        self.driver.find_element(By.ID, "newPassword").send_keys("NewPass123")
        confirm_input = self.driver.find_element(By.ID, "confirmPassword")
        confirm_input.send_keys("NewPass123")
        match_text = self.driver.find_element(By.ID, "passwordMatch")
        assert "match" in match_text.text
        assert "text-success" in match_text.get_attribute("class")

    def test_activity_stats_visibility(self):
        stats = ["totalPredictions", "averageGrade", "predictionFrequency", "firstPrediction", "lastPrediction"]
        for s in stats:
            element = self.driver.find_element(By.ID, s)
            assert element.is_displayed()

    def test_email_field_is_readonly(self):
        email_input = self.driver.find_element(By.ID, "email")
        assert email_input.get_attribute("disabled") == "true"

    def test_theme_toggle_on_profile(self):
        theme_icon = self.driver.find_element(By.ID, "themeIcon")
        current_theme = self.driver.find_element(By.TAG_NAME, "html").get_attribute("data-bs-theme")
        
        self.driver.find_element(By.CLASS_NAME, "theme-switcher").click()
        new_theme = self.driver.find_element(By.TAG_NAME, "html").get_attribute("data-bs-theme")
        
        assert current_theme != new_theme

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])