import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from base_test import BaseTest
import os
import time

class TestRecords(BaseTest):

    def setup_method(self):
        self.set_up()
        self.driver.get(self.base_url + "/login")
        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        self.driver.execute_script("arguments[0].click();", submit_btn)
        wait = WebDriverWait(self.driver, 15)
        wait.until(EC.url_to_be(self.base_url + "/"))
        self.driver.get(self.base_url + "/records")
        wait.until(EC.presence_of_element_located((By.ID, "recordsTableContainer")))

    def teardown_method(self):
        self.tear_down()

    def test_search_functionality(self):
        wait = WebDriverWait(self.driver, 10)
        search_input = wait.until(EC.presence_of_element_located((By.ID, "searchInput")))
        search_input.send_keys("Excellent")
        time.sleep(1)
        rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in rows:
            if "No Records" not in row.text:
                assert "Excellent" in row.text

    def test_level_filter(self):
        wait = WebDriverWait(self.driver, 10)
        filter_dropdown = Select(wait.until(EC.presence_of_element_located((By.ID, "levelFilter"))))
        filter_dropdown.select_by_value("At Risk")
        time.sleep(1)
        rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        for row in rows:
            if "No Records" not in row.text:
                assert "At Risk" in row.text

    def test_view_details_modal(self):
        wait = WebDriverWait(self.driver, 10)
        view_btns = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".action-btn.text-primary")))
        if len(view_btns) > 0:
            self.driver.execute_script("arguments[0].click();", view_btns[0])
            modal_title = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#detailsModal .modal-title")))
            assert "Details" in modal_title.text

    def test_export_csv_button(self):
        wait = WebDriverWait(self.driver, 10)
        export_btn = wait.until(EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Export')]")))
        assert export_btn.is_displayed()
        self.driver.execute_script("arguments[0].click();", export_btn)

    def test_pagination_exists(self):
        wait = WebDriverWait(self.driver, 10)
        pagination = wait.until(EC.presence_of_element_located((By.ID, "pagination")))
        assert pagination.is_displayed()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])