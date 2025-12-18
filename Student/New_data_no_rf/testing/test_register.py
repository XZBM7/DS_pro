import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestRegister(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/register")

    def teardown_method(self):
        self.tear_down()

    def click_submit_js(self):
        self.driver.execute_script("document.getElementById('registerForm').setAttribute('novalidate', 'novalidate');")
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)

    def fill_all_fields(self, name="Ibrahim Amr", user="ibrahim_2025", email="test@example.com", pwd="Pass1234", confirm="Pass1234"):
        self.driver.find_element(By.ID, "full_name").clear()
        self.driver.find_element(By.ID, "full_name").send_keys(name)
        self.driver.find_element(By.ID, "username").clear()
        self.driver.find_element(By.ID, "username").send_keys(user)
        self.driver.find_element(By.ID, "email").clear()
        self.driver.find_element(By.ID, "email").send_keys(email)
        self.driver.find_element(By.ID, "password").clear()
        self.driver.find_element(By.ID, "password").send_keys(pwd)
        self.driver.find_element(By.ID, "confirm_password").clear()
        self.driver.find_element(By.ID, "confirm_password").send_keys(confirm)

    def test_successful_registration(self):
        self.fill_all_fields(email="user" + str(time.time()) + "@test.com")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        success_alert = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "alert-success")))
        assert "Account created successfully" in success_alert.text

    def test_name_too_short(self):
        self.fill_all_fields(name="I")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        alert = wait.until(EC.presence_of_element_located((By.ID, "alertContainer")))
        assert "at least 2 characters" in alert.text or "letters and spaces" in alert.text

    def test_password_mismatch(self):
        self.fill_all_fields(pwd="Pass1234", confirm="WrongPass99")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        alert = wait.until(EC.presence_of_element_located((By.ID, "alertContainer")))
        assert "do not match" in alert.text

    def test_password_complexity(self):
        self.fill_all_fields(pwd="123456", confirm="123456")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        alert = wait.until(EC.presence_of_element_located((By.ID, "alertContainer")))
        assert "one lowercase letter, one uppercase letter, and one number" in alert.text

    def test_duplicate_user(self):
        self.fill_all_fields(email="ibrahim@gmail.com")
        self.click_submit_js()
        wait = WebDriverWait(self.driver, 10)
        alert = wait.until(EC.presence_of_element_located((By.ID, "alertContainer")))
        assert "already exists" in alert.text

if __name__ == "__main__":
    pytest.main([__file__])