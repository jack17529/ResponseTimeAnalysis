long startTime = System.currentTimeMillis();

driver.get("http://zyxware.com");

new WebDriverWait(driver, 10).until(ExpectedConditions.

presenceOfElementLocated(By.id("Calculate")));

long endTime = System.currentTimeMillis();

long totalTime = endTime - startTime;

System.out.println("Total Page Load Time: " + totalTime + "

milliseconds");