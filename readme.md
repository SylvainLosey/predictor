# Predictor

To better display what we built for our project, we decided to create a simple web app that predicts wether a review is positive or negative.

[Link to the app](https://whispering-bastion-28721.herokuapp.com/)

## Model

We selected our most accurate model for this experiment:

#### Logistic Regression
- Accuracy: 0.836
- Trained on 3000 movie reviews to limit its size

More details can be found in the [main project](https://github.com/SylvainLosey/DMML2019_Team_Orange).


## Technology

The web app is built using Flask.

```app.py``` is the web app itself, it loads the trained model ```model.pkl```  
```build.py``` is used to train the model along with ```clean.py```

The app is then hosted on a free [Heroku](https://www.heroku.com/home) instance.
