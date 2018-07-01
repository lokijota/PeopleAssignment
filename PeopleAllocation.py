# Databricks notebook source
# MAGIC %md # People to Customers allocation
# MAGIC 
# MAGIC The goal of this workbook is to implement the automatic assignment of people to customers/locations based in a set of criteria like distance, affinity to customer, etc. The base criteria is travel time.

# COMMAND ----------

# MAGIC %md # TO-DO
# MAGIC 
# MAGIC - Plan for exceptions in travel means when calculating travel times (public transport, car)
# MAGIC - Draw polygons according to preliminary assignments

# COMMAND ----------

# MAGIC %md ## Setup Connections and prepare data 
# MAGIC Sample data only used to show the format. The real data is on the auxiliary notebook.

# COMMAND ----------

import numpy as np
import pandas as pd

people = np.array([ #name, address, 0/1-role
                   ('name1', 'address1', 0),
                   ('name2', 'address2', 1),
                  ])

customers = np.array([ #name, address, 0/1/2 - tier
                     ('name1', 'address1', 0),
                     ('name2', 'address2', 2)
                   ])


alignments = np.array([ #person, customer
                        ('name1', 'name2', 'name1'),
                        ('name2', 'name2' )
                   ])

googlemaps_key = 'AI.....obtain from google maps.....'

azure_blobstorage_account = 'storage account name'
azure_blobstorage_container = 'container name'
azure_blobstorage_accesskey = 'storage account access key'

# COMMAND ----------

# MAGIC %run ./Setup

# COMMAND ----------

# MAGIC %md ## Mount Azure Storage
# MAGIC 
# MAGIC Mount Azure Storage in the DBFS filesystem. This will contain the UKMap, distances, and any other output

# COMMAND ----------

try:
  dbutils.fs.mount(
      source = "wasbs://"+azure_blobstorage_container+"@"+azure_blobstorage_account+".blob.core.windows.net/",
      mount_point = "/mnt/hipo",
      extra_configs = {"fs.azure.account.key."+azure_blobstorage_account+".blob.core.windows.net": azure_blobstorage_accesskey})
except:
    print("(Mount already existed, ignoring)")

# COMMAND ----------

# MAGIC %md ## Prepare data set
# MAGIC 
# MAGIC Make some changes to the variables defined above (in the Setup notebook) to facilitate processing

# COMMAND ----------

# add new columns for the lat/long
people = np.hstack((people, np.zeros((people.shape[0], 2), dtype=float))) 

# add new columns for the lat/long
customers = np.hstack((customers, np.zeros((customers.shape[0], 2), dtype=float))) # add new columns for the lat/long

# create a distances numpi array and convert to pandas dataframe
distances = np.zeros((np.shape(people)[0], np.shape(customers)[0]), dtype=float) #using float because I'll be aplying multipliers later
df_distances = pd.DataFrame(index=customers[:,0], columns=people[:,0])
df_distances = df_distances.fillna(0) # with 0s rather than NaNs

print("# of customers is", np.shape(customers)[0])
print("# of people is", np.shape(people)[0])

#print(customers[:,0])
#print(df_distances)

# COMMAND ----------

# MAGIC %md ## Obtain base distances (time to arrive in minutes)
# MAGIC 
# MAGIC Obtain travel time from people to locations and fill a distance matrix.
# MAGIC 
# MAGIC Python library to use google maps from here: https://github.com/googlemaps/google-maps-services-python
# MAGIC 
# MAGIC Steps:
# MAGIC - Key: variable googlemaps_key
# MAGIC - Added googlemaps library to the cluster
# MAGIC - Added several APIs (geocode, directions, distance matrix) in https://console.developers.google.com/google/maps-apis/api-list?project=your_project_name
# MAGIC - Used https://jsonformatter.curiousconcept.com/ to parse the json returned and understand the structure

# COMMAND ----------

import json
import googlemaps
import math
from datetime import datetime

gmap = googlemaps.Client(key=googlemaps_key)

# Returnt the distance in minutes between two locations
def distance_in_minutes(origin, destination, mode):
  tuesday0930 = datetime_object = datetime.strptime('Jul 19 2018  9:30AM', '%b %d %Y %I:%M%p') #datetime.now()
  directions_result = gmap.directions(origin, destination, mode=mode, arrival_time=tuesday0930)

  if(destination == 'Guernsey'):
      return 5*60 #assume 5 hours to get there

  try:
    g_departure_time = directions_result[0]['legs'][0]['departure_time']['text']
    g_arrival_time = directions_result[0]['legs'][0]['arrival_time']['text']
  except:
    print(directions_result[0])
    
  dif = datetime.strptime(g_arrival_time,"%I:%M%p") - datetime.strptime(g_departure_time,"%I:%M%p")
  return math.ceil(dif.seconds/60)

# Geocoding an address
# geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# Look up an address with reverse geocoding
# reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# Request directions via public transit

#print(distance_in_minutes("West Kensington, UK", "Paddington, UK", "transit"), "minutes")

# COMMAND ----------

# Change this to True if it's the first time running the code. After it a file will be written to DBFS and you can use it as a cache to avoid re-calling Google Maps all the time
recalculate_distances = False

if recalculate_distances == True:
  for indexp, person in enumerate(people):
    for indexc, customer in enumerate(customers):
      #if(distances[indexp][indexc] == 0):
      #distances[indexp][indexc] = distance_in_minutes(person[1], customer[1], "transit")
      if(df_distances.iloc[indexc, indexp] == 0):
        df_distances.iloc[indexc, indexp] = distance_in_minutes(person[1], customer[1], "transit")
      print(person[0], "to", customer[1], "takes", df_distances.iloc[indexc, indexp], "minutes")
  df_distances.to_json('/dbfs/mnt/hipo/df_distances.json', orient='split')
else:
  df_distances = pd.read_json('/dbfs/mnt/hipo/df_distances.json', orient='split')

print(df_distances)

# COMMAND ----------

# Get the Latitude and Longitude for all the people and Customers and write to the array
for person in people:
  geocode_result = gmap.geocode(person[1])
  person[3] = geocode_result[0]['geometry']['location']['lat']
  person[4] = geocode_result[0]['geometry']['location']['lng']
  print(person)

for customer in customers:
  geocode_result = gmap.geocode(customer[1])
  customer[3] = geocode_result[0]['geometry']['location']['lat']
  customer[4] = geocode_result[0]['geometry']['location']['lng']
  print(customer)

# COMMAND ----------

# MAGIC %md ## Explore data visually and get some statistics
# MAGIC 
# MAGIC Varied data exploration experiments.

# COMMAND ----------

print("Who's on average closer to the customers:")
print(df_distances.mean().sort_values())

print("What customers are on average closer:")
print(df_distances.mean(axis=1).sort_values())

# COMMAND ----------

# MAGIC %md ### Any-to-any distances with conditional formatting

# COMMAND ----------

# https://stackoverflow.com/questions/17748570/conditional-formatting-for-2-or-3-scale-coloring-of-cells-of-a-table
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = df_distances

## Exploration: Set all distances above a certain value to a maximum.
#df[df > 150] = 150

#print(df)
#vals = df.round(0) #df.values

normal = 1-(df - df.min()) / (df.max() - df.min()) #Normalize data to [0, 1] range for color mapping below

fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')

ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center', cellColours=plt.cm.RdYlGn(normal), animated=True)

ax.set_xlim(-0.5, 5.5)
fig.subplots_adjust(left=0.22)

display(fig)

# Save to file and copy to DBFS
fig.savefig("/tmp/table_cf.png")
dbutils.fs.cp("file:///tmp/table_cf.png", "/mnt/hipo/table_cf.png") 

## TODO: calculate improvement in travel distances (on average)

# COMMAND ----------

# MAGIC %md ### Closest N Customers per person

# COMMAND ----------

#https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
topN = 6
ind = np.argpartition(df_distances, topN, axis=0)[:topN]

closest_distances = df_distances.copy()

# ugly but didn't find another way to do it -- keep only the topN elements per column; +1000 is an inneficient trick
for y in range(0, ind.shape[1]):
  for x in range(0, ind.shape[0]):
    row_to_modify = ind.iloc[x,y]
    closest_distances.iloc[row_to_modify, y] = closest_distances.iloc[row_to_modify, y] + 1000
    
closest_distances[closest_distances < 1000] = np.nan
closest_distances -= 1000

#Normalize data to [0, 1] range for color mapping below
normal = 1-(closest_distances - closest_distances.min()) / (closest_distances.max() - closest_distances.min()) 
normal = (normal.notnull()).astype('float') / 2 #otherwise the maximum in the table gets very dark

closest_distances = closest_distances.replace(np.nan,'') # to avoid printint nan's

#and now generate the image
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')

ax.table(cellText=closest_distances.values, rowLabels=closest_distances.index, colLabels=closest_distances.columns, loc='center', cellColours=plt.cm.Greens(normal), animated=True)

ax.set_xlim(-0.5, 5.5)
fig.subplots_adjust(left=0.22)

display(fig)

# Save to file and copy to DBFS
fig.savefig("/tmp/table_closest_per_person.png")
dbutils.fs.cp("file:///tmp/table_closest_per_person.png", "/mnt/hipo/table_closest_per_person.png") 

# COMMAND ----------

# MAGIC %md ### Closest N Persons per Customer

# COMMAND ----------

#https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
topN = 3
ind = np.argpartition(df_distances, topN, axis=1)[:30]

closest_distances = df_distances.copy()

# ugly but didn't find another way to do it -- keep only the topN elements per row -- in this case we get the full table so the loop is simpler
for x in range(0, ind.shape[0]):
  for y in range(topN, ind.shape[1]):
    column_to_modify = ind.iloc[x,y]
    closest_distances.iloc[x, column_to_modify] = np.nan

#Normalize data to [0, 1] range for color mapping below
normal = 1-(closest_distances - closest_distances.min()) / (closest_distances.max() - closest_distances.min()) 
normal = (normal.notnull()).astype('float') / 2 #otherwise the maximum in the table gets very dark

closest_distances = closest_distances.replace(np.nan,'') # to avoid printint nan's

#and now generate the image
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')

ax.table(cellText=closest_distances.values, rowLabels=closest_distances.index, colLabels=closest_distances.columns, loc='center', cellColours=plt.cm.Greens(normal), animated=True)

ax.set_xlim(-0.5, 5.5)
fig.subplots_adjust(left=0.22)

display(fig)

# Save to file and copy to DBFS
fig.savefig("/tmp/table_closest_per_customer.png")
dbutils.fs.cp("file:///tmp/table_closest_per_customer.png", "/mnt/hipo/table_closest_per_customer.png") 

# COMMAND ----------

# MAGIC %md ## Mapped data
# MAGIC Explore using geographical maps to show information using gmap's APIs
# MAGIC 
# MAGIC https://github.com/pbugnion/gmaps --> Hangs databricks/had erratic behaviour (Python version problems?)
# MAGIC 
# MAGIC http://www.datadependence.com/2016/06/creating-map-visualisations-in-python/ --> mpl_toolkits.basemap not installed in mpl_toolkits
# MAGIC 
# MAGIC https://docs.databricks.com/user-guide/visualizations/charts-and-graphs-python.html --> native maps very limited
# MAGIC 
# MAGIC http://vincent.readthedocs.io/en/latest/quickstart.html#simple-map > doesn't show the graph (display doesn't work)
# MAGIC 
# MAGIC 
# MAGIC This seems to be the only remaning alternative (well, I could always use PowerBI, of course)
# MAGIC 
# MAGIC https://stackoverflow.com/questions/6999621/how-to-use-extent-in-matplotlib-pyplot-imshow
# MAGIC 
# MAGIC https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb
# MAGIC 
# MAGIC http://www.bigendiandata.com/2017-06-27-Mapping_in_Jupyter/

# COMMAND ----------

# MAGIC %md ### Copy an image with the UK Map from Azure Storage to the local cluster filesystem

# COMMAND ----------

#dbutils.fs.mkdirs("/mnt/hipo")

try:
  dbutils.fs.cp("/mnt/hipo/UKMap.png", "file:///tmp/UKMap.png") 
except :
  print("(some error while copying file to local /tmp)")

# Take a look at the file system just in case -- should have the UKMap.png and the CSV file with distances
display(dbutils.fs.ls("/mnt/hipo/"))

# COMMAND ----------

# MAGIC %md ### Plot on top of the image with the map

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Load an image
im = plt.imread('/tmp/UKMap.png')

# Set the alpha
alpha = 0.9

# Creare your figure and axes
fig,ax = plt.subplots(1)

# Set whitespace to 0

fig.set_size_inches(6.29*1.24,9.28*1.24)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

# Display the image
ax.imshow(im,alpha=alpha,extent=(-5.728, 1.79, 49.46, 56.27)) #49.39, 56.14

# Turn off axes and set axes limit
ax.axis('tight')
ax.axis('off')

# Plot the scatter points for people and for customers
role0 = people[ people[:,2] == '0' ]
ax.scatter(role0[:,4], role0[:,3],c="blue",s=7**2,linewidths=.2,alpha=.7, marker="v")

role1 = people[ people[:,2] == '1' ]
ax.scatter(role1[:,4], role1[:,3],c="blue",s=7**2,linewidths=.2,alpha=.7, marker="v")

customers_t1t2 = customers[ customers[:,2] != '3' ]
ax.scatter(customers_t1t2[:,4], customers_t1t2[:,3],c="darkgreen",s=7**2,linewidths=.2,alpha=.6)

customers_t3 = customers[ customers[:,2] == '3' ]
ax.scatter(customers_t3[:,4], customers_t3[:,3],c="lightgreen",s=7**2,linewidths=.2,alpha=.6)

#for i, txt in enumerate(people):
#  ax.annotate(people[i,0][:3], (people[i,4], people[i,3]))

for i, txt in enumerate(customers_t1t2):
  ax.annotate(customers_t1t2[i,0][:3].lower(), (customers_t1t2[i,4], customers_t1t2[i,3]),color='darkgreen', fontsize=12, style='italic')

  
#corners_sx = np.array([-5.728, -5.728, 1.79, 1.79])
#corners_sy = np.array([49.39, 56.14, 49.39, 56.14]) 
#ax.scatter(corners_sx, corners_sy,c="yellow",s=64,linewidths=1,alpha=1)

display(plt.show())

fig.savefig("/tmp/map_with_locations.png")
dbutils.fs.cp("file:///tmp/map_with_locations.png", "/mnt/hipo/map_with_locations.png") 

# COMMAND ----------

# MAGIC %md ## Calculate allocations
# MAGIC 
# MAGIC Use "polyamorous" stable marriage algorithm to populate allocations: Hospital-Residents / College Admissions problem

# COMMAND ----------

#person_distances = df_distances.get('ali')
#person_distances.index = range(num_customers)
#person_distances = person_distances.sort_values(ascending=True)

#print(person_distances)

# COMMAND ----------

#From customer list remove those from tier > 2
important_customers = []
distances_with_id = df_distances.copy()
for customer in customers:
  if int(customer[2]) < 3:
    important_customers.append(customer)
  else:
    name = customer[0]
    distances_with_id = distances_with_id.drop(name)
important_customers = np.asarray(important_customers)
num_customers = len(important_customers[:,0])
distances_with_id.index = range(num_customers)

# COMMAND ----------

##test for equal number of people and costumers (for simple stable marriage implementation)
##Order data according to travel time
num_people = len(people[:,0])
#num_customers = len(important_customers[:,0])

people_preferences = pd.DataFrame(index = range(num_customers), columns = people[:,0])
people_preferences = people_preferences.fillna(0)

#do a copy of distance table using customer ID instead of name for easier usage afterwards
#distances_with_id = df_distances.copy()
#distances_with_id.index = range(num_customers)

for indexp, person in enumerate(people):
  person_distances = distances_with_id.get(person[0])
  person_distances = person_distances.sort_values(ascending=True)
  people_preferences.iloc[:,indexp] = person_distances.keys()
  
##This part should be different but now it is random just to test implementation
#customer_preferences = pd.DataFrame(index = people[:,0], columns = customers[0:num_people,0])
#customer_preferences = customer_preferences.fillna(0)

#for indexc, customer in enumerate(customers[0:num_people]):
#  p = people[:,0]
#  np.random.shuffle(p)
#  customer_preferences.iloc[:,indexc] = p

##Encoding costumer preferences with values instead of order
customer_preferences = pd.DataFrame(index = people[:,0], columns = customers[:,0])
#put a 4 to test a way of testing a way of distributing the customers between all SAs
customer_preferences = customer_preferences.fillna(4)

print(people_preferences)
print(customer_preferences)

# COMMAND ----------

#-1 would mean the customer still hasn't been assigned
#Then element will be filled with the id of the person handling this customer
assigned_customers=(-1)*np.ones(num_customers)
#assigned_people = []

matches = [[] for i in range(num_people)]

while(np.any(assigned_customers==-1)):

  propositions = []
  
  ##each person proposes to her/his costumer
  for indexp, person in enumerate (people):
    #check if they have already 4 clients
    if len(matches[indexp]) < 4:
      
      i = 0
      first_available_customer = people_preferences.get(person[0])[i]

      while assigned_customers[first_available_customer]!=-1:
        i+=1
        #This is customer id
        first_available_customer = people_preferences.get(person[0])[i]

      propositions.append((first_available_customer,customers[first_available_customer][0],indexp,person[0]))

  ##Costumers have a chance to accept or reject the proposals based on their own preferences
  while(len(propositions) > 0):
    customer_id,customer,person_id,person = propositions.pop()
    
    #if this customer has still not been assigned just assign it to the person who proposed
    if assigned_customers[customer_id] == -1:
      matches[person_id].append(customer)
      assigned_customers[customer_id] = person_id
      #Because person is getting one more customer he is less desirable to the other ones
      customer_preferences.loc[person] = customer_preferences.loc[person] - 1 
    #  assigned_people.append(person)
      
    #if it has been assigned it can accept or reject the proposition
    else:
      matched_person = int(assigned_customers[customer_id])
      pref = customer_preferences.get(customer)

   #   matched_pref = pref.get(pref == matched_person.get_values()[0]).index[0]
    #Changed to preferences based on value, instead of order
      matched_pref = pref.get(matched_person)

     # new_pref = pref.get(pref == person).index[0]
      new_pref = pref.get(person)
      
      #If customer prefers new SA to the one that's already matched, change the pair, otherwise do nothing
      if(new_pref > matched_pref):

        matches[matched_person].remove(customer)
        matches[person_id].append(customer)
        assigned_customers[customer_id] = person_id
        
        #So the person who has "lost" this client is more free for others and the one that is now asigned is less desirable
        matched_name = people[matched_person][0]
        customer_preferences.loc[matched_name] = customer_preferences.loc[matched_name] + 1 
        customer_preferences.loc[person] = customer_preferences.loc[person] - 1 
        
        ##assigned_people.remove(matched_person)
       ## assigned_people.append(person)


      
#Just print results
for i in range(num_people):
  print(people[i][0])
  print(matches[i])
