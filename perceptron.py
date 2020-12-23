import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # для того щоб представити текстові дані у вигляді вектора
from sklearn.metrics.pairwise import cosine_similarity  # для cosine similarity
import difflib  # для пошуку максимально схожих слів
import numpy as np
from math import e
import tkinter
from tkinter.ttk import *
from tkinter import simpledialog
import tkinter.messagebox
import random
import webbrowser

processed = False 
def callback(url):
    webbrowser.open_new(url)
    
def exit_window():
    """
    Функція для виходу з програми
    """
    root.destroy()
    raise SystemExit

def clear_scr(screen):
    """
    Функція для видалення з екрану елементів графічного інтерфейсу
    """

    for wid in screen:
        wid.grid_forget()


def logistische_f(alpha,sigma):
    
    f = 1/(1+e**(-alpha*sigma))
    
    return f

def delta_finden(y_modell,y_erwartet):

    delta = abs((y_modell-y_erwartet)/y_erwartet)

    return delta


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def title_from_index(index,movie):
    '''
        Допоміжна функція яка дістає
        тайтл фільму за стовпчиком index
    '''
    return movie[movie.index == index]["original_title"].values[0]

def index_from_title(original_title,movie):
    '''
        Допоміжна функція яка дістає
        index фільму за стовпчиком original_title
    '''
    title_list = movie['original_title'].tolist()
    common = difflib.get_close_matches(original_title, title_list, 1)  # шукаємо найближчу схожу назву
    titlesim = common[0]
    return movie[movie.original_title == titlesim]["index"].values[0]

def combine_features(row):
    '''
        Допоміжна функція для того, щоб
        створити єдиний рядок який містить
        слова з усіх факторів
    '''
    try:
        return row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row['director']+" "+row['tagline']
    except:
        print ("Error:", row)

        
def cos_s_train(df):
    movie = df
    movie["index"] = [i for i in range(len(movie))]
    movie

    features = ['keywords','cast','genres','director','tagline'] # Виділяємо необхідні фактори для порівняння

    for feature in features:
        movie[feature] = movie[feature].fillna('') # Заповнюємо всі пропущені дані порожніми рядками

    # за допомогою методу **apply** застосовуємо функцію
    # **combine_features** для кожного рядку і формуємо новий стовпчик в датасеті

    movie["combined_features"] = movie.apply(combine_features,axis=1)

    # ### Загальний опис того що відбувається далі
    # 1. **fit_transform** - ця функція трансформує кожен рядок в вектор, при цьому враховуючи всі інші слова в стопці датасеті, при цьому виходить дуже довгий вектор з 0 та 1.

    cv = CountVectorizer()  # Furious 7
    count_matrix = cv.fit_transform(movie["combined_features"])  # це щось типу onehot
    cosine_sim = cosine_similarity(count_matrix)

    return cosine_sim, movie


def cos_s_use(cosine_sim, movie, user_movie):

    movie_index = index_from_title(user_movie, movie)

    similar_movies =  list(enumerate(cosine_sim[movie_index]))  # дістаємо вектор з значеннями від [0, 1] номеруємо їх як в index
    similar_movies_sorted = sorted(similar_movies,  key=lambda x:x[1],  reverse=True)  # сортуємо за значенням косинуса

    result = []
    
    i=0
    for rec_movie in similar_movies_sorted:
            if(i != 0):
                result.append(title_from_index(rec_movie[0],movie))
            i=i+1
            if i > 50:
                break

    return result


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def fav_percent(target_string, fav_list):
    """
        Допоміжна функція для отримання відсотку улюблених акторів/жанрів/директорів у рядку (target_string)
    """
    percentage = 0
    for favourite in fav_list:
        if favourite.lower() in target_string.lower():
            percentage += 1/len(fav_list)

    if percentage>1:
        percentage = 1
    return percentage

def pre_proc(df, fav_cast, fav_dirs, fav_genres):
    """
        Допоміжна функція для виділення важливих ознак.
        fav_cast, fav_dirs, fav_genres - люблені актори, директори та жанри, відповідно
        df - датафрейм
    """

    #useful_columns = ['popularity', 'budget', 'revenue', 'original_title','cast',
    #              'director','runtime', 'genres',
    #              'production_companies', 'vote_average', 'release_year']

    useful_columns = ['popularity', 'budget', 'revenue', 'original_title','cast',
                  'director', 'runtime', 'genres', 'vote_average']

    # Подумати над нормальним word embedding
    # keywords , що часто зустрічаються -- це ще одна фіча ?
    # production_companies ?
    
    df2 = df[useful_columns].fillna('') # Беремо корисні стовпчики

    # Word embedding для самих
    global processed
    if processed == False:
        df2['director'] = df2['director'].apply(lambda director: fav_percent(director, fav_dirs))
        df2['cast'] = df2['cast'].apply(lambda cast: fav_percent(cast, fav_cast))
        df2['genres'] = df2['genres'].apply(lambda genre: fav_percent(genre, fav_genres))
        processed = True


    # Нормалізація стовпчиків з великиими значеннями
    for column in ['budget', 'revenue','runtime']:
        x_min = df2[column].min()
        x_max = df2[column].max()
        df2[column] = (df2[column] - x_min)/(x_max-x_min)
        
    return df2




def perceptron_online(x_list, y_list, epsilon, alpha, width):
    global progress
    global progress_var
    # Структура: len(x_list[0]) - width - width - 1

    W1 = np.random.uniform(-2,2,(width, len(x_list[0])))
    B1 = np.random.uniform(0.1,4,(width,1))

    W2 = np.random.uniform(-2,2,(width,width))
    B2 = np.random.uniform(0.1,4,(width,1)) 

    W3 = np.random.uniform(-2,2,(1,width))
    B3 = np.array([[round(random.uniform(0.1,4),3)]]).transpose()

    delta = [0 for i in range(len(x_list))]

    for epoch in range(1001):
        for i in range(len(x_list)):
            x_beispiel = np.array([x_list[i]]).transpose()
            y_beispiel = y_list[i]

            #print(x_beispiel, y_beispiel)
            y_2 = logistische_f(alpha,B1 + np.dot(W1,x_beispiel))
            y_3 = logistische_f(alpha,B2 + np.dot(W2,y_2))
            y_4 = logistische_f(alpha,B3 + np.dot(W3,y_3))

          
            d_4 = alpha*y_4*(1-y_4)*(y_beispiel-y_4)
            
            d_3 = np.dot(W3.transpose(),d_4)
            d_3 = np.array([alpha*d_3[i]*(1-y_3[i])*y_3[i] for i in range(len(d_3))])
            

            d_2 = np.dot(W2.transpose(),d_3)
            d_2 = np.array([alpha*d_2[i]*(1-y_2[i])*y_2[i] for i in range(len(d_2))])

            B3 += d_4
            B2 += d_3
            B1 += d_2


            W3 += np.dot(y_3,d_4).transpose()
            """
            for j in range(len(W3)):
                for k in range(len(W3[0])):
                    W3[j][k] += y_3[k][0]*d_4[j][0]
            """
            for j in range(len(W2)):
                for k in range(len(W2[0])):
                    W2[j][k] += y_2[k][0]*d_3[j][0]
            for j in range(len(W1)):
                for k in range(len(W1[0])):
                    W1[j][k] += x_beispiel[k][0]*d_2[j][0]

            
            y_2 = logistische_f(alpha,B1 + np.dot(W1,x_beispiel))
            y_3 = logistische_f(alpha,B2 + np.dot(W2,y_2))
            y_4 = logistische_f(alpha,B3 + np.dot(W3,y_3))

            delta[i] = round(float(delta_finden(y_4,y_beispiel)),5)

        progress_var.set(epoch)
        root.update_idletasks()
            
        avg_delta = np.average(delta)

        """
        if epoch%100 == 0:
            #print('\n----------')
            print('Номер епохи:',epoch+1,"\nПохибка:",avg_delta)
        """
            
        if avg_delta <= epsilon or epoch == 1000:
            f = open("weights.py","w+")
            f.write("import numpy as np \n")
            f.write("W1=np.array("+str(W1.tolist())+')\n')
            f.write("W2=np.array("+str(W2.tolist())+')\n')
            f.write("W3=np.array("+str(W3.tolist())+')\n')
            f.write("B1=np.array("+str(B1.tolist())+')\n')
            f.write("B2=np.array("+str(B2.tolist())+')\n')
            f.write("B3=np.array("+str(B3.tolist())+')\n')
            f.close()
            #print('Ваги успішно збережено у файл')
            break

already_seen = []
labels2 = []
def recognition(B1,B2,B3,W1,W2,W3,df,alpha):
    global progress
    global progress_var
    global labels2
    
    try:
        clear_scr(labels2)
    except:
        pass

    global already_seen

    


    progress_var.set(1050)
    
    def feedforvard(x_beispiel):
        x_beispiel = np.array([x_beispiel]).transpose()
        y_2 = logistische_f(alpha,B1 + np.dot(W1,x_beispiel))
        y_3 = logistische_f(alpha,B2 + np.dot(W2,y_2))
        y_4 = logistische_f(alpha,B3 + np.dot(W3,y_3))

        return y_4[0][0]*5

    progress_var.set(1100)
    
    df['predict'] = df.apply(lambda row: feedforvard(row.drop(['original_title']).to_numpy()),axis = 1)

    df = df.sort_values(by=['predict'],ascending = False).dropna()

    count = df[['original_title', 'predict']].where((df['predict']>=2.5) & (~df['original_title'].isin(already_seen))).dropna().shape[0]

    if count>=5:
        result = df[['original_title', 'predict']].where((df['predict']>=2.5) & (~df['original_title'].isin(already_seen))).dropna().iloc[:5]
    else:
        result = df[['original_title', 'predict']].where((df['predict']>=2.5) & (~df['original_title'].isin(already_seen))).dropna().iloc[:count]

    if result.empty:
        result = df[['original_title', 'predict']].where(~df['original_title'].isin(already_seen)).dropna().iloc[:5]
        
    ik = 0
    for index, row in result.iterrows():
        already_seen.append(row['original_title'])
        exec(f"label_{ik} = tkinter.Label(root,text=\'{result['original_title'].loc[index]}\',font=('Arial Black', 9))")
        exec(f"label_{ik}.grid(row = 7+ik, column = 0, columnspan = 2,padx = 25)")
        exec(f"labels2.append(label_{ik})")
        exec(f"label_{ik}.bind('<Button-1>', lambda e: callback('https://www.google.com/search?q=imdb+{result['original_title'].loc[index]}'))")

        exec(f"global but_{ik}1; but_{ik}1 = tkinter.Button(text = 'Подобається', command = lambda: add_film('{result['original_title'].loc[index]}', 4.95),font=('Times New Roman',11))")
        exec(f"but_{ik}1.grid(row = 7+ik, column = 3,pady = 10)")
        exec(f"labels2.append(but_{ik}1)")

        exec(f"global but_{ik}2; but_{ik}2 = tkinter.Button(text = 'Не подобається', command = lambda: add_film('{result['original_title'].loc[index]}', 0.05),font=('Times New Roman',11))")
        exec(f"but_{ik}2.grid(row = 7+ik, column = 4,pady = 10)")
        exec(f"labels2.append(but_{ik}2)")
        
        ik += 1

    global but_next5
    but_next5 = tkinter.Button(text = 'Створити нові рекомендації', command = recommend,font=("Times New Roman",11))
    but_next5.grid(row = 9+ik, column = 0, columnspan = 4,pady = 10)
    labels2.append(but_next5)
    
        

def separate_unique(df_col):
    df_col = df_col.dropna()
    list_unique = set()
    def helper_sep(string):
        helper = string.split('|')
        for el in helper:
            list_unique.add(el)

    df_col.apply(helper_sep)

    return list_unique


        
def add_actor(actor_name):
    global cast
    global fav_cast
    user_cast = actor_name
    u_c = cast.where(cast.str.contains(user_cast)).dropna()
    if u_c.empty:
        tkinter.messagebox.showinfo(title='Результат', message="На жаль, такого актора не знайдено, спробуйте ще раз")
        return -1
    elif (len(u_c) !=1):
        u_c = u_c.reset_index()
        del u_c['index']
        inx = simpledialog.askstring(title = "Результат", prompt = "Знайдено декілька співпадінь!  \n" + str(u_c)+"\nУведіть код того, кого ви мали на увазі:")
        inx = int(inx)
        if u_c.iloc[inx].iloc[0] not in fav_cast:
            fav_cast.append(u_c.iloc[inx].iloc[0])
            tkinter.messagebox.showinfo(title='Результат', message="Додано актора "+str(u_c.iloc[inx].iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цього актора")
        return -1
    elif (len(u_c) == 1):
        if u_c.iloc[0] not in fav_cast:
            fav_cast.append(u_c.iloc[0])
            tkinter.messagebox.showinfo(title='Результат', message="Додано актора "+str(u_c.iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цього актора")
        return -1

def add_director(dir_name):
    global dirs
    global fav_dirs
    user_dirs = dir_name
    u_d = dirs.where(dirs.str.contains(user_dirs)).dropna()
    if u_d.empty:
        tkinter.messagebox.showinfo(title='Результат', message="На жаль, такого режисера не знайдено, спробуйте ще раз")
        return -1
    elif (len(u_d) !=1):
        u_d = u_d.reset_index()
        del u_d['index']
        inx = simpledialog.askstring(title = "Результат", prompt = "Знайдено декілька співпадінь!  \n" + str(u_d)+"\nУведіть код того, кого ви мали на увазі:")
        inx = int(inx)
        if u_d.iloc[inx].iloc[0] not in fav_dirs:
            fav_dirs.append(u_d.iloc[inx].iloc[0])
            tkinter.messagebox.showinfo(title='Результат', message="Додано режисера "+str(u_d.iloc[inx].iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цього режисера")
        return -1
    elif (len(u_d) == 1):
        if u_d.iloc[0] not in fav_dirs:
            fav_dirs.append(u_d.iloc[0])
            tkinter.messagebox.showinfo(title='Результат', message="Додано режисера "+str(u_d.iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цього режисера")
        return -1

def add_genre(genre_name):
    global genres
    global fav_genres
    user_genres = genre_name
    u_g = genres.where(genres.str.contains(user_genres)).dropna()
    if u_g.empty:
        tkinter.messagebox.showinfo(title='Результат', message="На жаль, такого жанру не знайдено, спробуйте ще раз")
        return -1
    elif (len(u_g) !=1):
        u_g = u_g.reset_index()
        del u_g['index']
        inx = simpledialog.askstring(title = "Результат", prompt = "Знайдено декілька співпадінь!  \n" + str(u_g)+"\nУведіть код того жанру, якого ви мали на увазі:")
        inx = int(inx)
        if u_g.iloc[inx].iloc[0] not in fav_genres:
            fav_genres.append(u_g.iloc[inx].iloc[0])
            tkinter.messagebox.showinfo(title='Результат', message="Додано жанр "+str(u_g.iloc[inx].iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цей жанр")
        return -1
    elif (len(u_g) == 1):
        if u_g.iloc[0] not in fav_genres:
            fav_genres.append(u_g.iloc[0])
            tkinter.messagebox.showinfo(title='Результат', message="Додано жанр "+str(u_g.iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цей жанр")
        return -1

def add_film(film, mark, prompt_mode = True):
    global batya_rating
    batya_film = film
    if prompt_mode == True:
        found_films = df['original_title'].where(df['original_title'].str.contains(batya_film)).dropna()
    if prompt_mode == False:
        found_films = df['original_title'].where(df['original_title'] == batya_film).dropna()
    if found_films.empty:
        tkinter.messagebox.showinfo(title='Результат', message="На жаль, такого фільму не знайдено, спробуйте ще раз")
    elif (len(found_films) !=1):
        found_films = found_films.reset_index()
        del found_films['index']
        inx = simpledialog.askstring(title = "Результат", prompt = "Знайдено декілька співпадінь!  \n" + str(found_films)+"\nУведіть код того фільму, якого ви мали на увазі:")
        inx = int(inx)
        if found_films.iloc[inx].iloc[0] not in batya_rating.keys():
            batya_rating[found_films.iloc[inx].iloc[0]] = mark/5
            tkinter.messagebox.showinfo(title='Результат', message="Обрано фільм "+str(found_films.iloc[inx].iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цей фільм")
    elif (len(found_films) == 1):
        if found_films.iloc[0] not in batya_rating.keys():
            batya_rating[found_films.iloc[0]] = mark/5
            tkinter.messagebox.showinfo(title='Результат', message="Обрано фільм "+str(found_films.iloc[0]))
        else:
            tkinter.messagebox.showinfo(title='Результат', message="Ви вже відмітили цей фільм")
            
def directors_view():
    global labels

    try:
        clear_scr(labels)
    except:
        pass

    label_dirs = tkinter.Label(root,text='Введіть декількох улюблених режисерів \n(англійською мовою, по одному імені)',font=("Arial Black", 9))
    label_dirs.grid(row = 2, column = 0, columnspan = 4,padx = 25)
    labels.append(label_dirs)

    dirs_e = tkinter.Entry(root, width = 35, borderwidth = 5)
    dirs_e.grid(row = 3, column = 0, columnspan = 4, pady = 10)
    labels.append(dirs_e)

    but_dirs = tkinter.Button(text = 'Додати', command = lambda: add_director(dirs_e.get()),font=("Times New Roman",11))
    but_dirs.grid(row = 4, column = 0, columnspan = 4,pady = 10)
    labels.append(but_dirs)

    but_next2 = tkinter.Button(text = 'Далі', command = genres_view,font=("Times New Roman",11))
    but_next2.grid(row = 5, column = 0, columnspan = 4,pady = 10)
    labels.append(but_next2)


def genres_view():
    global labels

    try:
        clear_scr(labels)
    except:
        pass

    
    label_genres = tkinter.Label(root,text='Введіть декілька улюблених жанрів',font=("Arial Black", 9))
    label_genres.grid(row = 2, column = 0, columnspan = 4,padx = 25)
    labels.append(label_genres)

    combo_e = Combobox(root, values = genres.tolist(),state="readonly")
    combo_e.grid(row = 3, column = 0, columnspan = 4, pady = 10)
    combo_e.current(0)
    labels.append(combo_e)

    but_g = tkinter.Button(text = 'Додати', command = lambda: add_genre(combo_e.get()),font=("Times New Roman",11))
    but_g.grid(row = 4, column = 0, columnspan = 4,pady = 10)
    labels.append(but_g)

    but_next3 = tkinter.Button(text = 'Далі', command = films_view1,font=("Times New Roman",11))
    but_next3.grid(row = 5, column = 0, columnspan = 4,pady = 10)
    labels.append(but_next3)

def films_view1():
    global labels

    try:
        clear_scr(labels)
    except:
        pass
    
    label_film1 = tkinter.Label(root,text='Введіть декілька фільмів, які Вам подобаються \n(англійською мовою, по одній назві)',font=("Arial Black", 9))
    label_film1.grid(row = 2, column = 0, columnspan = 4,padx = 25)
    labels.append(label_film1)

    film1_e = tkinter.Entry(root, width = 35, borderwidth = 5)
    film1_e.grid(row = 3, column = 0, columnspan = 4, pady = 10)
    labels.append(film1_e)

    but_f1 = tkinter.Button(text = 'Додати', command = lambda: add_film(film1_e.get(),4.95),font=("Times New Roman",11))
    but_f1.grid(row = 4, column = 0, columnspan = 4,pady = 10)
    labels.append(but_f1)

    but_next4 = tkinter.Button(text = 'Далі', command = films_view2,font=("Times New Roman",11))
    but_next4.grid(row = 5, column = 0, columnspan = 4,pady = 10)
    labels.append(but_next4)

def films_view2():
    global batya_rating
    if len(batya_rating) == 0:
        tkinter.messagebox.showerror(title='Результат', message="Введіть принаймні 1 фільм")
        return -1
    
    global labels

    try:
        clear_scr(labels)
    except:
        pass

    label_film2 = tkinter.Label(root,text='Введіть декілька фільмів, які Вам НЕ подобаються \n(англійською мовою, по одній назві)',font=("Arial Black", 9))
    label_film2.grid(row = 2, column = 0, columnspan = 4,padx = 25)
    labels.append(label_film2)

    film2_e = tkinter.Entry(root, width = 35, borderwidth = 5)
    film2_e.grid(row = 3, column = 0, columnspan = 4, pady = 10)
    labels.append(film2_e)

    but_f2 = tkinter.Button(text = 'Додати', command = lambda: add_film(film2_e.get(),0.05),font=("Times New Roman",11))
    but_f2.grid(row = 4, column = 0, columnspan = 4,pady = 10)
    labels.append(but_f2)

    global but_next5
    but_next5 = tkinter.Button(text = 'Створити рекомендації', command = lambda: recommend(),font=("Times New Roman",11))
    but_next5.grid(row = 5, column = 0, columnspan = 4,pady = 10)
    labels.append(but_next5)



def recommend():
    global batya_rating
    global cosine_sim
    global movie
    global cos_movies
    
    if len(batya_rating) < 2:
        tkinter.messagebox.showerror(title='Результат', message="Введіть принаймні ще 1 фільм")
        return -1

    clear_scr([but_next5])
    root.update()

    
    global progress
    global progress_var
    progress_var = tkinter.DoubleVar()
    progress = Progressbar(root, orient = "horizontal", length = 100, variable=progress_var,maximum = 1100)
    progress.grid(row = 6, column = 0, columnspan = 4,pady = 10)
    labels.append(progress)
    
    global df

    df = pre_proc(df, fav_cast, fav_dirs, fav_genres) # Попередня обробка

    for user_movie in batya_rating.keys():
        cos_movies += cos_s_use(cosine_sim,movie, user_movie)
    cos_movies = list(set(cos_movies))

    x_list = df.where(df['original_title'].isin(batya_rating.keys())).dropna() # Обираємо лише ті фільми, які оцінив наш батя
    x_list['target'] = x_list['original_title'].map(batya_rating)
    y_list = x_list['target'].to_numpy()

    del x_list['target']
    del x_list['original_title']
    
    x_list = x_list.to_numpy()

    epsilon = 0.1
    alpha = 0.7
    width = 8
    
    perceptron_online(x_list, y_list, epsilon, alpha, width) # Навчання (ваги зберігаються у файл)
    
    from weights import W1,W2,W3,B1,B2,B3

    df1 = df.where(df['original_title'].isin(cos_movies)).dropna()
    
    recognition(B1,B2,B3,W1,W2,W3,df1, alpha)    
    
if __name__ == '__main__':

    global df
    df = pd.read_csv('movies.csv') #імпортуємо дані
    #Типовий батуа:
    global fav_cast; fav_cast = [] #'Vin Diesel','Jason Statham'
    global fav_dirs; fav_dirs = [] #'Miller','Tarantino'
    global fav_genres; fav_genres = [] #'Action','Crime'
    global batya_rating; batya_rating = {} #'Furious 7':5/5, 'Terminator Genisys': 5/5, 'Mad Max: Fury Road': 5/5,'Titanic': 0.1/5,'Spider-Man 3': 0.1/5, 'Big Hero 6': 0.1/5
    
    global cast; cast = pd.Series(list(separate_unique(df['cast'])))
    global dirs; dirs = pd.Series(list(separate_unique(df['director'])))
    global genres; genres = pd.Series(list(separate_unique(df['genres'])))

    global cosine_sim
    global movie
    cosine_sim, movie = cos_s_train(df)

    global cos_movies
    cos_movies = []

    
    root = tkinter.Tk()
    root.title('Система рекомендації фільмів')
    root.geometry('540x550')

    labels = []

    label_start1 = tkinter.Label(root,text='● Система рекомендації фільмів',font=("Arial Black", 9))
    label_start1.grid(row = 1, column = 0, columnspan = 4,padx = 25, pady = 20)


    label_acts = tkinter.Label(root,text='Введіть декількох улюблених акторів \n(англійською мовою, по одному імені)',font=("Arial Black", 9))
    label_acts.grid(row = 2, column = 0, columnspan = 4,padx = 25)
    labels.append(label_acts)

    acts_e = tkinter.Entry(root, width = 35, borderwidth = 5)
    acts_e.grid(row = 3, column = 0, columnspan = 4, pady = 10)
    labels.append(acts_e)

    but_acts = tkinter.Button(text = 'Додати', command = lambda: add_actor(acts_e.get()),font=("Times New Roman",11))
    but_acts.grid(row = 4, column = 0, columnspan = 4,pady = 10)
    labels.append(but_acts)

    but_next1 = tkinter.Button(text = 'Далі', command = directors_view,font=("Times New Roman",11))
    but_next1.grid(row = 5, column = 0, columnspan = 4,pady = 10)
    labels.append(but_next1)

    """
    but_exit = tkinter.Button(text = 'Вийти', command = exit_window, bg = 'red', fg = 'white',font=("Arial Black", 8))
    but_exit.grid(row = 100, column = 0, columnspan = 100, pady = 25)
    """
    root.mainloop() 

