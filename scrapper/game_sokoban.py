import requests
from bs4 import BeautifulSoup
import json

# Define the range of IDs
ids = range(925, 1079 + 1)

# Define the base URL
base_url = "http://www.game-sokoban.com/index.php?mode=level_info&ulid={index}&view=general"

# Initialize a dictionary to store the results
results = {}

# Iterate over each ID in the range
for id in ids:
    url = base_url.format(index=id)
    try:
        # Make the request
        response = requests.get(url)
        
        # Check if the response is successful
        if response.status_code == 200:
            # Parse the HTML content with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all rows with the class 'row'
            rows = soup.find_all('div', class_='row')

            # Dictionary to store the extracted data for the current level
            level_data = {}

            for row in rows:
                header = row.find('div', class_='header')
                content = row.find('div', class_='content')

                if header and content:
                    # Use the header text as the key and the content text as the value
                    header_text = header.get_text(strip=True)
                    content_text = content.get_text(strip=True)
                    
                    # Handle preformatted content correctly (e.g. for the 'Code' field)
                    if content.find('pre'):
                        content_text = content.find('pre').get_text(strip=True)
                    
                    # Store the header-content pair in the level_data dictionary
                    level_data[header_text] = content_text
            print(level_data)
            # Store the level data in the results dictionary
            results[id] = level_data
        else:
            print(f"Failed to fetch data for ID {id}: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while fetching data for ID {id}: {str(e)}")

# Optionally, save the results to a file
with open("scrapper/sokoban_level_data.json", "w") as outfile:
    json.dump(results, outfile, indent=4)

# If you want to print the results to check:
print(json.dumps(results, indent=4))
