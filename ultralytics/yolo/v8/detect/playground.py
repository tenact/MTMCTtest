import os

height, width, _ = img.shape

file_path = "data.json.txt"

if os.path.isfile(file_path): 
        # if the file exists, read its contents
        with open(file_path, 'r') as f:
            #if os.path.getsize(file_path) != 0:
            contents = f.read()
            for line in contents.split('\n'):
                if not line.strip():
                    continue  # skip empty lines
                data = json.loads(line)
                i = data['number']
                arr = data['array']

                # do the ReID
                #Iteration über die JSON-File und vergleich mit key

            # Loop over each line in the text file
            

                # Access the number and array from the JSON data
            

                # Do something with the number and array (for example, print them)
                print('Number:', i)
                print('Array:', arr)
                
                reid = REID()

                if(reid.euclidian_distance(reid.extract_features(sub_image), arr) > 0.7):
                    print("Similarity: ", reid.euclidian_distance(reid.extract_features(sub_image), arr))
                
                addIdToJsonFile = False
                newID = i
                #update the ID
                #and do not add the ID to the list
                #end the loop

                # hier muss auch noch die schwarze box gezeichnet werden    


                break

            if(addIdToJsonFile):
                
                # Save the modified JSON data to a new file
                with open('new_file.txt', 'w+') as f: # before 'new_file.txt'
                    # Write the JSON data to a new line in the file
                    f.write(json.dumps({'number': identities[i], 'array':reid.extract_feature(sub_image)}) + '\n')

    
else:
# if the file does not exist, create a new one
    with open(file_path, 'w') as f:
        print("File created.")        
        new_data = {'number': key, 'array': reid.extract_features(sub_image)}
        f.write(json.dumps(new_data) + '\n')
    # and add all the content of identities


# wir iterieren über alle objekte rüber (oftmals nur eins)


# wir bekommen die Daten des DeepSorts als Ergebnis

# schreiben der Ergebnisse in eine Datei
# dabei prüfen, ob diese Daten bereits existieren
    #ReID und dann die Darstellung des Bounding Box ändern