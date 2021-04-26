# With this magic you'll save the content of the cell into a file named "cluster_app.py"

import streamlit as st # We list every library that we must import
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv') # For what I have in mind I'll need the data
model = pickle.load(open('model.pkl', 'rb')) # We load the trained model

st.write("""
# Let's cluster some numbers 😎 
""") # This is the main title (check the image!)

st.sidebar.header("User input's") # This is the sidebar title

def user_input_features(): # We define functions to read the user inputs over the streamlit sliders controls and return them as a pandas dataframe
    x_ = st.sidebar.slider('x value', 1, 1000, int(dataset['x'].mean())) # We establish the range 1-1000 for the inputs and set the default in the mean of the value in the train set
    y_ = st.sidebar.slider('y value', 1, 1000, int(dataset['y'].mean())) # Same as above!
    data = {'y': y_,
            'x': x_,
            'data_type': 'prediction'}
            
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features() # We retrieve the input using the function defined above and save it as "df"

st.subheader('Your x & y values:') # This is the title of the next print (the df)

st.write(df) # We print the data just to show the selected values in case the user closes the sidebar


cluster_n = model.predict(df[['x', 'y']]) # We are saving the predicted label for the user input, you'll see why!


temp_df = pd.concat([dataset[['x', 'y']], df], axis= 0) # Now we put the input and the train dataset together
temp_df['cluster'] = model.predict(temp_df[['x', 'y']]) # Add a column named 'cluster'
temp_df.fillna('train', inplace=True) # This is to fill the "data_type" field and distinguish between training and user input instances

st.subheader('# of cluster') # Just a title

st.write(cluster_n ) # Now we do use the predicted cluster of the input we stored before

st.write(temp_df) # We print the full dataset (check the interactivity, neat!)

st.title("Scatterplot") # Now we add a title for the scatterplot

fig, ax = plt.subplots() # First define the scatterplot
ax =sns.scatterplot(data=temp_df,
                    x="x",
                    y="y",
                    hue="cluster", # Color by cluster
                    style="data_type", # To drawn different shapes
                    size='data_type', # We're going to set some difference in the size also
                    sizes={'prediction': 100, # You can just state the sizes in a dictionary
                            'train': 20})

ax.legend(bbox_to_anchor=(1, 1), # This is to move the legends outside the plot
           borderaxespad=1)

st.pyplot(fig) # Finally, we render this scatterplot in streamlit 
