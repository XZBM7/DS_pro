from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class BaseTest:
    driver = None
    base_url = "http://127.0.0.1:5000"

    def set_up(self):
        chrome_driver_path = r"C:\ChromeDriver\chromedriver-win64\chromedriver.exe"
        
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--remote-allow-origins=*")
        options.add_argument("--disable-notifications")
        
        service = Service(executable_path=chrome_driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.implicitly_wait(10)

    def tear_down(self):
        if self.driver:
            self.driver.quit()