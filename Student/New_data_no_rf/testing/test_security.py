import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestSecurity(BaseTest):

    def setup_method(self):
        self.set_up()

    def teardown_method(self):
        self.tear_down()

    def test_unauthorized_access_redirect(self):
        self.driver.get(self.base_url + "/dashboard")
        wait = WebDriverWait(self.driver, 10)
        error_title = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "error-title")))
        assert "Access Denied" in error_title.text

    def test_login_button_redirection(self):
        self.driver.get(self.base_url + "/dashboard")
        wait = WebDriverWait(self.driver, 10)
        login_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Sign In")))
        login_btn.click()
        assert "/login" in self.driver.current_url

    def test_register_button_redirection(self):
        self.driver.get(self.base_url + "/dashboard")
        wait = WebDriverWait(self.driver, 10)
        register_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Create Account")))
        register_btn.click()
        assert "/register" in self.driver.current_url

    def test_auto_redirect_if_logged_in(self):
        self.driver.get(self.base_url + "/login")
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        
        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))
        
        self.driver.get(self.base_url + "/dashboard") 
        
        time.sleep(2)
        
        current_url = self.driver.current_url
        assert current_url == self.base_url + "/" or "/dashboard" in current_url

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])