import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest
import time

class TestDashboard(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)

        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))
        
        self.driver.get(self.base_url + "/dashboard")
        wait.until(EC.presence_of_element_located((By.ID, "welcomeName")))

    def teardown_method(self):
        self.tear_down()

    def test_stats_cards_display(self):
        wait = WebDriverWait(self.driver, 10)
        total = wait.until(EC.presence_of_element_located((By.ID, "statTotal")))
        assert total.is_displayed()

    def test_recent_predictions_table(self):
        wait = WebDriverWait(self.driver, 10)
        table_body = wait.until(EC.presence_of_element_located((By.ID, "recentTableBody")))
        rows = table_body.find_elements(By.TAG_NAME, "tr")
        if len(rows) > 0 and "Loading" not in rows[0].text:
            assert len(rows) <= 5

    def test_quick_action_navigation(self):
        nav_link = self.driver.find_element(By.XPATH, "//a[contains(@href, '/')]")
        self.driver.execute_script("arguments[0].click();", nav_link)
        assert self.driver.current_url == self.base_url + "/"

    def test_dashboard_delete_modal(self):
        wait = WebDriverWait(self.driver, 10)
        time.sleep(2) 
        delete_btns = self.driver.find_elements(By.CLASS_NAME, "action-btn")
        if len(delete_btns) > 0:
            self.driver.execute_script("arguments[0].click();", delete_btns[0])
            modal = wait.until(EC.visibility_of_element_located((By.ID, "deleteModal")))
            assert "Delete" in modal.text

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])