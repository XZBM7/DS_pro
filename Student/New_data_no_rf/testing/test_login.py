import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestLogin(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")

    def teardown_method(self):
        self.tear_down()

    def click_submit_js(self):
        self.driver.execute_script("document.getElementById('loginForm').setAttribute('novalidate', 'novalidate');")
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)

    def fill_login_form(self, user="ibrahim_2025", pwd="Pass1234"):
        self.driver.find_element(By.ID, "username").clear()
        self.driver.find_element(By.ID, "username").send_keys(user)
        self.driver.find_element(By.ID, "password").clear()
        self.driver.find_element(By.ID, "password").send_keys(pwd)

    def test_successful_login(self):
        self.fill_login_form(user="ibrahim_2025", pwd="Pass1234")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        success_alert = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "alert-success")))
        assert "success" in success_alert.text.lower() or "login" in success_alert.text.lower()
        time.sleep(1)
        assert self.driver.current_url == self.base_url + "/"

    def test_invalid_username_format(self):
        self.fill_login_form(user="ib", pwd="Pass1234")
        feedback = self.driver.find_element(By.ID, "usernameFeedback")
        assert "at least 3 characters" in feedback.text

    def test_wrong_credentials_api(self):
        self.fill_login_form(user="wrong_user", pwd="WrongPassword123")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        error_alert = wait.until(EC.visibility_of_element_located((By.ID, "alertContainer")))
        assert "Invalid" in error_alert.text or "failed" in error_alert.text.lower()

    def test_empty_fields_validation(self):
        self.driver.find_element(By.ID, "username").clear()
        self.driver.find_element(By.ID, "password").clear()
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        alert = wait.until(EC.visibility_of_element_located((By.ID, "alertContainer")))
        assert "fill" in alert.text.lower() or "required" in alert.text.lower()

    def test_theme_toggle_login(self):
        theme_btn = self.driver.find_element(By.CLASS_NAME, "theme-switcher")
        theme_btn.click()
        html_tag = self.driver.find_element(By.TAG_NAME, "html")
        assert html_tag.get_attribute("data-bs-theme") == "dark"

if __name__ == "__main__":
    pytest.main([__file__])