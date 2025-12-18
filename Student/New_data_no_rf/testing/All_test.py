import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_test import BaseTest


class TestModelVisualsPage(BaseTest):

    def setup_method(self):
        self.set_up()
        self.login()
        self.driver.get(self.base_url + "/model_visuals")

    def teardown_method(self):
        self.tear_down()

    def login(self):
        self.driver.get(self.base_url + "/login")

        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )

        self.driver.find_element(By.ID, "username").send_keys("ibrahim_2025")
        self.driver.find_element(By.ID, "password").send_keys("Pass1234")
        self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        WebDriverWait(self.driver, 20).until(
            EC.url_contains("/")
        )

   
    def test_page_header_loaded(self):
        header = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.TAG_NAME, "h2"))
        )
        assert header.text.strip() == "Model Visualizations"

   
    def test_all_charts_rendered(self):
        wait = WebDriverWait(self.driver, 30)

        charts = [
            "lossChart",
            "predActualChart",
            "featureImpChart",
            "residualsChart"
        ]

        for chart_id in charts:
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, f"#{chart_id} .main-svg")
                )
            )

            svg = self.driver.find_element(
                By.CSS_SELECTOR, f"#{chart_id} .main-svg"
            )
            assert svg.is_displayed()

   
    def test_loading_spinners_hidden(self):
        wait = WebDriverWait(self.driver, 30)

        wait.until(
            EC.invisibility_of_element_located(
                (By.CLASS_NAME, "loading")
            )
        )

    def test_theme_toggle(self):
        html = self.driver.find_element(By.TAG_NAME, "html")
        initial_theme = html.get_attribute("data-bs-theme")

        theme_btn = self.driver.find_element(By.ID, "themeToggle")
        theme_btn.click()

        WebDriverWait(self.driver, 5).until(
            lambda d: html.get_attribute("data-bs-theme") != initial_theme
        )

        new_theme = html.get_attribute("data-bs-theme")
        assert new_theme in ["light", "dark"]
        assert new_theme != initial_theme

   
    def test_navbar_links(self):
        links = {
            "/analytics": "//a[contains(@href,'/analytics')]",
            "/dashboard": "//a[contains(@href,'/dashboard')]",
            "/profile": "//a[contains(@href,'/profile')]",
        }

        for path, xpath in links.items():
            link = self.driver.find_element(By.XPATH, xpath)
            self.driver.execute_script("arguments[0].click();", link)

            WebDriverWait(self.driver, 10).until(
                EC.url_contains(path)
            )

            self.driver.back()


if __name__ == "__main__":
    pytest.main(["-v", "-s"])
