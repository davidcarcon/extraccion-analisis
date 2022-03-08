from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import tweepy
import pandas as pd
import re
import plotly.express as px
import networkx as nx
import datetime
import nltk
from emotion import saca_score
from googletrans import Translator
#nltk.download('punkt')


st.set_option('deprecation.showPyplotGlobalUse', False)

if "component_cleared" not in st.session_state:
    st.session_state["component_cleared"] = False

if "component_cleared_a" not in st.session_state:
	st.session_state["component_cleared_a"] = True

stopwords = open('spanish.txt')
stopwords = set(map(lambda x: x.replace('\n', ''), stopwords.readlines()))

translator = Translator()

@st.cache
def convert_df(dataframe):
	return dataframe.to_csv().encode('utf-8')

def aplica_func(textos):
	enojo = []
	alegria = []
	optimismo = []
	tristeza = []
	for texto in textos:
		text_en = translator.translate(texto, dest='en').text
		dd = saca_score(text_en)
		enojo.append(dd[0])
		alegria.append(dd[1])
		optimismo.append(dd[2])
		tristeza.append(dd[3])
	return [enojo, alegria, optimismo, tristeza]

def quita_pipe(cadena):
	salida = cadena.replace('|', ' ').replace('\n', '')
	return salida

def tipoLemaFrec(listFrec, palabra):
	salida = []
	for k, i in listFrec:
		if k[0].startswith(palabra[:5]):
			cambio = palabra 
			salida.append((cambio, k[1], i))
		elif k[1].startswith(palabra[:5]):
			cambio = palabra
			salida.append((k[0], cambio, i))
	salida = sorted(salida, key=lambda x: x[2], reverse=True)[:10]
	salida = list(map(lambda x: (x[0], x[1]), salida))
	return salida

def extrae_entidades_hashtags(dataframe):
	mi_dic = {}
	user = dataframe.Usuario_twitter.unique()
	for i in user:
		mi_dic[i] = []
		teu = dataframe[dataframe['Usuario_twitter'] == i]
		text = ' '.join(teu['Texto'].to_list())
		for x in text.split():
			if x.startswith('@') or x.startswith('#'):
				mi_dic[i].append(x)
	return mi_dic


def nubes_usuario(usuario, textos):
	texto = para_nube(textos, stopwords)
	wordcloud = WordCloud().generate(texto)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()
	container = st.container()
	container.markdown("**{}**".format(usuario))
	container.pyplot()
	return container

def para_nube(textos, set_stop):
	aux = []
	textos = textos.split()
	for i in textos:
		if not(i.startswith('@')) and not(i.startswith('https')):
			aux.append(i)
	texto = " ".join(aux)
	tuitf = re.sub(r'[^\w\s]', '', texto).lower()
	tuitf = list(set(tuitf.split()) - stopwords)
	tuitf = " ".join(tuitf)
	return tuitf


def cambio_username():
	st.session_state.username = False

def cambio_concepto():
	st.session_state.concepto = False

def button_click_callback():
	st.session_state["component_cleared"] = False
	st.session_state["component_cleared_a"] = True

def button_submmit_callback():
    st.session_state["component_cleared"] = True
    st.session_state["component_cleared_a"] = False
    api = oauth_login()
    if not st.session_state["component_cleared_a"]:
    	if st.session_state.concepto == True:
    		st.title('Análisis de Tweets :exclamation::exclamation:')
    		num = st.session_state.numero
    		conceptos =  list(map(lambda x: x.strip(), st.session_state.texto.split(',')))
    		mod = st.session_state.modo
    		df, dfu = extrae_tuits_concepto(api, conceptos, num, mod)
    		lista_emociones = aplica_func(df.Texto.to_list())
    		df['enojo'] = lista_emociones[0]
    		df['alegría'] = lista_emociones[1]
    		df['optimismo'] = lista_emociones[2]
    		df['tristeza'] = lista_emociones[3]
    		st.table(df.sample(10))
    		st.header('Usuarios con mayor numero de followers')
    		dfu = dfu.drop_duplicates().sort_values(by='Numero Followers', ascending=False).head(5)
    		fig = px.bar(x=dfu['Numero Followers'], y=dfu['Usuario Twitter'],  
    					orientation='h',
    					 labels={
    					 	"x": "Número de Followers",
    					 	"y": "UserName en Twitter"
    					 })
    		fig.update_traces(showlegend=False)
    		st.write(fig)
    		usua = [dfu.iloc[0]['Nombre'], dfu.iloc[1]['Nombre'], dfu.iloc[2]['Nombre'], 
    				dfu.iloc[3]['Nombre'], dfu.iloc[4]['Nombre']]
    		desc = [quita_pipe(dfu.iloc[0]['Descripcion']), quita_pipe(dfu.iloc[1]['Descripcion']), 
    				quita_pipe(dfu.iloc[2]['Descripcion']), quita_pipe(dfu.iloc[3]['Descripcion']), 
    				quita_pipe(dfu.iloc[4]['Descripcion'])]
    		data = {
    			'Usuarios': usua,
    			'Descripcion': desc
    		}
    		st.header('Información de los Usuarios')
    		df_aux = pd.DataFrame(data, columns=['Usuarios', 'Descripcion'])
    		st.table(df_aux)

    		st.header('Palabras al rededor de las busquedas')
    		buscarU = []
    		textosU = " ".join(df.Texto.to_list())
    		textosU = para_nube(textosU, stopwords)
    		tokens = nltk.word_tokenize(textosU)
    		bgs = nltk.bigrams(tokens)
    		fdist = nltk.FreqDist(bgs)
    		for option in conceptos:
    			aux = []
    			for k,v in fdist.items():
    				if (k[0].startswith(option[:5])) or (k[1].startswith(option[:5])):
    					aux.append((k, v))
    			aux = tipoLemaFrec(aux, option)
    			buscarU += aux
    		G = nx.Graph()
    		G.add_edges_from(buscarU)
    		fig, ax = plt.subplots(figsize=(15, 8))
    		pos = nx.spring_layout(G, k=0.4)
    		nx.draw_networkx(G, pos,
    			font_size=10,
    			width=3,
    			edge_color='grey',
    			node_color='purple',
    			with_labels = False,
    			ax=ax)
    		for key, value in pos.items():
    			x, y = value[0]+.08, value[1]+.035
    			ax.text(x, y,
    				s=key,
    				bbox=dict(facecolor='white', alpha=0.2),
    				horizontalalignment='center', fontsize=11)
    		st.pyplot(fig)
    		csv = convert_df(df)
    		st.download_button(
    			label="Descarga archivo csv e Inicia otra búsqueda",
    			data=csv,
    			file_name='conceptos.csv',
    			mime='text/csv',
    			on_click=button_click_callback
    		)
    	else:
    		usernames =  list(map(lambda x: x.strip(), st.session_state.usernames.split(',')))
    		numu = st.session_state.numero_u
    		usuarios, textos = extrae_guarda_tuits(api, usernames, numu)
    		us = textos.Usuario_twitter.unique()
    		st.header('Información de los Usuarios :adult:')
    		st.table(usuarios)
    		st.header('Muestra de los ultmos tuits de los usuarios :astonished:')
    		st.table(textos.sample(10))
    		lista_emociones_u = aplica_func(textos.Texto.to_list())
    		textos['enojo'] = lista_emociones_u[0]
    		textos['alegría'] = lista_emociones_u[1]
    		textos['optimismo'] = lista_emociones_u[2]
    		textos['tristeza'] = lista_emociones_u[3]
    		tuitst = ' '.join(textos['Texto'].to_list())
    		salida = para_nube(tuitst, stopwords)
    		st.header('Nube de palabras de todos los tuits extraidos :cloud:')
    		wordcloud = WordCloud().generate(salida)
    		plt.imshow(wordcloud, interpolation='bilinear')
    		plt.axis("off")
    		plt.show()
    		st.pyplot()
    		st.header('Nube de palabras por cada usuario de la busqueda :cloud:')
    		col = st.columns(4)
    		for i, user in enumerate(us):
    			textos_usuarios = textos[textos['Usuario_twitter'] == user]
    			tt = ' '.join(textos_usuarios['Texto'].to_list())
    			with col[i]:
    				nubes_usuario(user, tt)
    		st.header('Entidades y Hasthgs que menciona cada usuario')
    		json = extrae_entidades_hashtags(textos)
    		st.json(json)
    		csvt = convert_df(textos)
    		st.download_button(
    			label="Descarga archivo csv e Inicia otra búsqueda",
    			data=csvt,
    			file_name='textos_usuarios.csv',
    			mime='text/csv',
    			on_click=button_click_callback
    		)

def oauth_login():
	''' 
	Acceso a API twitter
	'''
	consumer_key = st.secrets['consumer_key']
	consumer_secret = st.secrets['consumer_secret']
	access_token = st.secrets['access_token']
	access_token_secret = st.secrets['access_token_secret']	
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	return api

def extrae_tuits_concepto(acceso, listaConceptos, cuantos, tipo):
	'''
	Busca tuits por conceptos
	'''
	dfusuariospro = pd.DataFrame()
	dftotal = pd.DataFrame()
	for concepto in listaConceptos:
		for status in tweepy.Cursor(acceso.search_tweets, q=' ' + concepto + ' ', 
									geocode='19.42847,-99.12766,800km',
									lang='es', result_type=tipo,
									include_entities=False,
									tweet_mode='extended').items(cuantos):
			registro = {
				'Fecha_creacion': pd.to_datetime(datetime.datetime.strptime(status._json['created_at'], 
					'%a %b %d %H:%M:%S %z %Y').date()),
				'Usuario_twitter': status._json['user']['screen_name'],
				'Texto': status._json["full_text"],
			}
			registro_u = {
				'Nombre': status._json['user']['name'],
				'Usuario Twitter': status._json['user']['screen_name'],
				'Descripcion': status._json['user']['description'],
				'Numero Followers': status._json['user']['followers_count'],
				'Localidad': status._json['user']['location'],
			}
			dftotal = dftotal.append(registro, ignore_index=True)
			dfusuariospro = dfusuariospro.append(registro_u, ignore_index=True) 
	return dftotal, dfusuariospro

def extrae_guarda_tuits(acceso, listaConceptos, cuantos):
	'''
	Busca tuits con los username
	'''
	dftuits = pd.DataFrame()
	dfusers = pd.DataFrame()
	for concepto in listaConceptos:
		status_u = acceso.get_user(id=concepto)
		registro_usuario = {
			'Nombre': str(status_u._json["name"]),
			'Usuario_twitter': str(status_u._json["screen_name"]),
			'Ubicacion': str(status_u._json["location"]),
			'Descripcion': str(status_u._json["description"].replace('\n', '')),
			'Numero_followers': str(status_u._json["followers_count"])
		}
		dfusers = dfusers.append(registro_usuario, ignore_index=True)
		for status in tweepy.Cursor(acceso.user_timeline, id=concepto,
									tweet_mode="extended").items(cuantos):
			registro = {
				'Fecha_creacion': pd.to_datetime(datetime.datetime.strptime(status._json['created_at'],
					'%a %b %d %H:%M:%S %z %Y').date() ),
				'Usuario_twitter': status._json['user']['screen_name'],
				'Texto': status._json["full_text"]
			}
			dftuits = dftuits.append(registro, ignore_index=True)
	return dfusers, dftuits

def crear_form():
	st.title('Extracción de Tweets :exclamation::exclamation:')
	username = st.checkbox('UserNames', key='username', on_change=cambio_concepto)
	concepto = st.checkbox('Conceptos', key='concepto', on_change=cambio_username)
	if concepto:
		formulario = st.form('mi_form')
		formulario.number_input('Número de Tweets por concepto', min_value=10, 
								 max_value=100, key='numero')
		formulario.selectbox('Que tipo de tweets se quiere descargar',
							('Mixed', 'Recent', 'Popular'), key='modo')
		formulario.text_area('Conceptos para buscar separados por ,', key='texto')
		formulario.form_submit_button('Ejecutar', on_click=button_submmit_callback)
		return formulario
	elif username:
		formulario_u = st.form('mi_form_1')
		formulario_u.number_input('Número de Tweets por username', min_value=10, 
								 max_value=100, key='numero_u')
		formulario_u.text_area('Usuarios de Twitter separados por ,', key='usernames')
		formulario_u.form_submit_button('Ejecutar', on_click=button_submmit_callback)
		return formulario_u

if not st.session_state["component_cleared"]:
	crear_form()



