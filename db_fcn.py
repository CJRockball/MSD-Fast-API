# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:23:20 2021

@author: PatCa
"""

import sqlite3



def db_action(db_command, db_params="", one_line=False, db_name='music_data.db'):
    #Open db connection
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    
    #Run action
    cur.execute(db_command, db_params)
    if one_line:
        result = cur.fetchone()
    else:
        result = cur.fetchall()
    
    # Write change
    conn.commit()
    conn.close()
    return result

def get_genre_sample(submit_genre):
    # Get more from the same genre
    sample_genre = ((submit_genre),)
    db_command = """SELECT title_table.title, song_data.confidence \
                 FROM song_data \
                     LEFT JOIN title_table \
                         ON song_data.song_id = title_table.song_id \
                 WHERE genre = ? LIMIT(10);"""
    song_samples = db_action(db_command=db_command, db_params=sample_genre, one_line=False)
    song_dict = [{'Song title': i[0], 'Confidence':i[1]} for i in song_samples]
    return song_dict

def get_db_sample():
    # Get songs from db
    db_command = """SELECT title_table.title,genre_table.genre_name, song_data.confidence \
                 FROM song_data \
                     LEFT JOIN title_table \
                         ON song_data.song_id = title_table.song_id \
                            LEFT JOIN genre_table \
                                ON song_data.genre = genre_table.genre_id \
                 ORDER BY song_data.song_id DESC \
                     LIMIT(10);"""
    song_samples = db_action(db_command=db_command, db_params="", one_line=False) #returns list of tuples with song title
    song_dict = [{'Song title': i[0], 'Genre':i[1], 'Confidence':i[2]} for i in song_samples]
    return song_dict

    #Put songs in list for easy handling
    song_list = []
    for i in song_samples:
        song_list.append(i[0])   
    return song_list

def check_song_in_db(song_name):
    param = ((song_name),)
    db_command = """Select title FROM title_table WHERE title = ?;"""
    name_check = db_action(db_command=db_command, db_params=param, one_line=True)        
    if name_check is None:
        return False
    else: return True
    
    
def nr_songs_in_db():
    #Check if song already in the db
    db_command = "Select count(title) FROM title_table;"
    num_titles = db_action(db_command, one_line=True)
    return  num_titles[0]    

def get_db_genre(song_name):
    param = ((song_name),)
    #Get genre and genre number
    db_command = """SELECT genre_table.genre_name, song_data.genre, song_data.confidence\
            FROM title_table \
                LEFT JOIN song_data \
                    ON title_table.song_id = song_data.song_id \
                        LEFT JOIN genre_table \
                            ON song_data.genre = genre_table.genre_id \
                        WHERE title_table.title = ?;"""
    genre_data = db_action(db_command, db_params=param, one_line=True)    
    return genre_data     

def load_song_to_db(song_title, tempo, duration, genre:int, confidence):
        title_tuple = ((song_title),)
        #Open database
        conn = sqlite3.connect("music_data.db")
        cur = conn.cursor()    
        #Inset new records
        cur.execute("INSERT OR IGNORE INTO title_table (title) VALUES (?);""",title_tuple)
        # Write change
        conn.commit()
        conn.close()      
        
        #Open database
        conn = sqlite3.connect("music_data.db")
        cur = conn.cursor()
        cur.execute("""SELECT song_id FROM title_table WHERE title = ?;""", title_tuple)
        new_song_id = cur.fetchone()
        result_tuple = (new_song_id[0], tempo, duration, genre, confidence)
        #Inset new records
        cur.execute("INSERT OR IGNORE INTO song_data (song_id,tempo,duration,genre,confidence) VALUES (?,?,?,?,?);""",
                        result_tuple)
        # Write change
        conn.commit()
        conn.close()
        return
























