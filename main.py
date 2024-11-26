import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color, Canvas
from kivy.properties import ListProperty, BooleanProperty, ObjectProperty
from kivy.metrics import dp
from kivy.clock import Clock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage
from PIL.ImageOps import flip
import io
import sympy as sp
import scipy as sci

if True:
	rook=sp.KroneckerProduct(sp.ones(8,8),sp.eye(8))+sp.KroneckerProduct(sp.eye(8),sp.ones(8,8))-sp.eye(64)*16
	bishop=sp.banded(64,{0:-2})+sp.FunctionMatrix(64,64,'lambda i,j: KroneckerDelta((j//8)-(i//8),(j%8)-(i%8))+KroneckerDelta((i//8)-(j//8),(j%8)-(i%8))').as_explicit()
	bishop=bishop-sp.diag(*(bishop*sp.ones(64,1)))
	queen=bishop+rook
	king=sp.KroneckerProduct(sp.banded(8,{0:1,1:1,-1:1}),sp.banded(8,{0:1,1:1,-1:1}))-sp.eye(64)
	king=king-sp.diag(*(king*sp.ones(64,1)))
	knight=sp.FunctionMatrix(64,64,'lambda i,j: KroneckerDelta(Abs((j//8)-(i//8))+Abs((j%8)-(i%8)),3)-KroneckerDelta(Abs((i//8)-(j//8)),3)*KroneckerDelta(j%8,i%8)-KroneckerDelta(3,Abs((j%8)-(i%8)))*KroneckerDelta(j//8,i//8)').as_explicit()
	knight=knight-sp.diag(*knight*sp.ones(64,1))
	whitepawn=sp.banded(64,{0:-1,8:1})
	blackpawn=sp.banded(64,{0:-1,-8:1})
	blackpawnatk=sp.banded(64,{0:-2})+sp.FunctionMatrix(64,64,'lambda i,j: KroneckerDelta((j//8)-(i//8),(j%8)-(i%8))*KroneckerDelta((j//8)-(i//8),-1)+KroneckerDelta((i//8)-(j//8),(j%8)-(i%8))*KroneckerDelta((j//8)-(i//8),-1)').as_explicit()
	whitepawnatk=sp.banded(64,{0:-2})+sp.FunctionMatrix(64,64,'lambda i,j: KroneckerDelta((j//8)-(i//8),(j%8)-(i%8))*KroneckerDelta((j//8)-(i//8),1)+KroneckerDelta((i//8)-(j//8),(j%8)-(i%8))*KroneckerDelta((j//8)-(i//8),1)').as_explicit()
	znorth=sp.banded(64,{8:1})
	zsouth=sp.banded(64,{-8:1})
	zeast=sp.Matrix(sp.KroneckerProduct(sp.eye(8),sp.banded(8,{-1:1})))
	zwest=sp.Matrix(sp.KroneckerProduct(sp.eye(8),sp.banded(8,{1:1})))
	znortheast=sp.Matrix(sp.KroneckerProduct(sp.banded(8,{1:1}),sp.banded(8,{-1:1})))
	znorthwest=sp.Matrix(sp.KroneckerProduct(sp.banded(8,{1:1}),sp.banded(8,{1:1})))
	zsoutheast=sp.Matrix(sp.KroneckerProduct(sp.banded(8,{-1:1}),sp.banded(8,{-1:1})))
	zsouthwest=sp.Matrix(sp.KroneckerProduct(sp.banded(8,{-1:1}),sp.banded(8,{1:1})))
	
	nqueen=np.array(queen) #Laplacian
	zqueen=sp.Matrix(np.multiply(np.greater(nqueen,0),nqueen)) #adjacency matrix
	nknight=np.array(knight)
	zknight=sp.Matrix(np.multiply(np.greater(nknight,0),nknight))
	nbishop=np.array(bishop)
	zbishop=sp.Matrix(np.multiply(np.greater(nbishop,0),nbishop))
	nking=np.array(king)
	zking=sp.Matrix(np.multiply(np.greater(nking,0),nking))
	nrook=np.array(rook)
	zrook=sp.Matrix(np.multiply(np.greater(nrook,0),nrook))
	nblackpawn=np.array(blackpawn)
	zblackpawn=sp.Matrix(np.multiply(np.greater(nblackpawn,0),nblackpawn))
	nwhitepawn=np.array(whitepawn)
	zwhitepawn=sp.Matrix(np.multiply(np.greater(nwhitepawn,0),nwhitepawn))
	nblackpawnatk=np.array(blackpawnatk)
	zblackpawnatk=sp.Matrix(np.multiply(np.greater(nblackpawnatk,0),nblackpawnatk))
	nwhitepawnatk=np.array(whitepawnatk)
	zwhitepawnatk=sp.Matrix(np.multiply(np.greater(nwhitepawnatk,0),nwhitepawnatk))
	piece_moveset={
		'queen': (7,znorth,zeast,zsouth,zwest,znortheast,znorthwest,zsouthwest,zsoutheast),
		'knight': (1,zknight),
		'bishop': (7,znortheast,znorthwest,zsouthwest,zsoutheast),
		'king': (1,znorth,zeast,zsouth,zwest,znortheast,znorthwest,zsouthwest,zsoutheast),
		'rook': (7,znorth,zeast,zsouth,zwest),
		'zqueen': zqueen,
		'zknight': zknight,
		'zbishop': zbishop,
		'zking': zking,
		'zrook': zrook,
		'blackpawn': (1,zblackpawn),
		'whitepawn': (1,zwhitepawn),
		'blackpawnatk': zblackpawnatk,
		'whitepawnatk': zwhitepawnatk
	}
	
	"""
	print(sum([zknight**i for i in range(2)],0*sp.eye(64)))
	print(A:=np.array(sum([(zknight*0.8018288963**0)**i for i in range(3)],0*sp.eye(64))*np.array(sp.FunctionMatrix(64,1,'lambda i,j: KroneckerDelta(i,63)').as_explicit()),dtype="int").reshape((8,8)))
	plt.imshow(A/np.argmax(A),cmap='hot',interpolation="nearest")
	plt.show()"""

class DraggablePiece(Image):
    def __init__(self, piece_type, color, board_ref, is_template=False, **kwargs):
        super().__init__(**kwargs)
        self.piece_type = piece_type
        self.piece_color = color
        #self.source = f'/data/user/0/ru.iiec.pydroid3/files/chess_pieces/{color}_{piece_type}.png'
        self.source=f'/storage/emulated/0/PydroidCode/chess_pieces/{color}_{piece_type}.png'
        self.size_hint = (None, None)
        self.size = (dp(50), dp(50))
        self.dragging = False
        self.original_pos = self.pos
        self.board_ref = board_ref
        self.current_square = None
        self.is_template = is_template

    def create_clone(self):
        clone = DraggablePiece(self.piece_type, self.piece_color, self.board_ref)
        clone.size = self.size
        clone.pos = self.pos
        return clone

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and not self.board_ref.locked:
            if self.is_template:
                # Create and start dragging a clone
                clone = self.create_clone()
                self.board_ref.add_widget(clone)
                clone.dragging = True
                touch.grab(clone)
                clone.on_touch_move(touch)
                return True
            else:
                self.dragging = True
                touch.grab(self)
                if self.current_square:
                    self.current_square.occupied_piece = None
                return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self and self.dragging:
            self.center = touch.pos
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self and self.dragging:
            self.dragging = False
            touch.ungrab(self)
            square = self.board_ref.find_square(touch.pos)
            
            if square:
                if square.occupied_piece:
                    # Remove the piece currently occupying this square
                    self.parent.remove_widget(square.occupied_piece)
                
                # Move to new square
                if self.current_square:
                    self.current_square.occupied_piece = None
                self.center = square.center
                square.occupied_piece = self
                self.current_square = square
            else:
                # Remove the piece if it's dropped outside of any square
                self.parent.remove_widget(self)
            return True
        return super().on_touch_up(touch)


class ChessSquare(Widget):
    def __init__(self, position, color, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.square_color = color
        self.size_hint = (None, None)
        self.size = (dp(50), dp(50))
        self.S=[]
        self.occupied_piece = None
        self.bind(pos=self.draw_square, size=self.draw_square)

    def draw_square(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.square_color)
            Rectangle(pos=self.pos, size=self.size)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and self.parent.locked and self.occupied_piece:
            self.parent.piece_clicked(self.occupied_piece)
            return True
        return super().on_touch_down(touch)

class ChessBoard(Widget):
    locked = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.squares = {}
        self.heatmap_texture = None
        self.state_vector = None
        self.enemy_territory_vector=None
        self.single_move_attack_vector=None
        self.rider_attack_vector=None
        self.position_vector=None
        self.adjacency=None
        self.heat_operator=None
        self.heatwaveclock=None
        self.quantum=False
        self.bind(pos=self.update_board, size=self.update_board)
        self.setup_board()

    def setup_board(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)
            #Rectangle(pos=self.pos, size=self.size)
            Window.clearcolor=(1,1,1,1)
        
        files = 'abcdefgh'
        for rank in range(8):
            for file in range(8):
                color = (0.9, 0.9, 0.9, 1) if (rank + file) % 2 == 0 else (0.5, 0.5, 0.5, 1)
                position = f'{files[file]}{8-rank}'
                square = ChessSquare(position=position, color=color)
                square.pos = (self.pos[0] + file * dp(50), 
                            self.pos[1] + (7-rank) * dp(50))
                self.squares[position] = square
                self.add_widget(square)

    def update_board(self, *args):
        for file in range(8):
            for rank in range(8):
                square = self.squares[f'{"abcdefgh"[file]}{8-rank}']
                square.pos = (self.pos[0] + file * dp(50),
                            self.pos[1] + (7-rank) * dp(50))

    def find_square(self, pos):
        for square in self.squares.values():
            if square.collide_point(*pos):
                return square
        return None

    def set_state_vector(self, clicked_piece):
        self.state_vector = np.zeros(64, dtype=int)
        self.enemy_territory_vector=np.zeros(64,dtype=int)
        self.single_move_attack_vector=np.ones(64,dtype=int)
        self.rider_attack_vector=np.ones(64,dtype=int)
        self.position_vector = np.zeros(64,dtype=int)
        enemy_riders=[]
        for rank in range(8):
            for file in range(8):
                pos = f'{"abcdefgh"[file]}{8-rank}'
                square = self.squares[pos]
                idx = rank * 8 + file
                if not square.occupied_piece or square.occupied_piece == clicked_piece:
                    self.state_vector[idx] = 1
                elif square.occupied_piece.piece_color != clicked_piece.piece_color:
                    self.enemy_territory_vector[idx] = 1
                    if square.occupied_piece.piece_type in ("rook","bishop","queen"):
                    	enemy_riders.append((square.occupied_piece,idx))
                    else:
                    	if square.occupied_piece.piece_type == "pawn":
                    		self.single_move_attack_vector*=1-(np.array(piece_moveset[f"{square.occupied_piece.piece_color}pawnatk"],dtype="int")@(np.eye(64,dtype=int)[idx]))
                    	else:
                    		self.single_move_attack_vector*=1-np.array(piece_moveset[f"z{square.occupied_piece.piece_type}"],dtype="int")@(np.eye(64,dtype=int)[idx])
                if square.occupied_piece == clicked_piece:
                	self.position_vector[idx]=1
        for square in enemy_riders:
            piece=square[0]
            idx=square[1]
            state=np.diag(self.state_vector)
            moveset=piece_moveset[piece.piece_type]
            for rider in moveset[1:]:
            	self.S=[np.eye(64,dtype="int")]
            	collider=state@np.array(rider,dtype="int")
            	taker=np.diag(self.enemy_territory_vector)@np.array(rider,dtype=int)
            	self.rider_attack_vector*=1-((np.eye(64,dtype=int)+taker)@(np.eye(64,dtype=int)+sum([self.memoize(self.S[0]@collider) for i in range(moveset[0])],0*np.eye(64,dtype="int")))-np.eye(64,dtype=int))@(np.eye(64,dtype=int)[idx])
        return

    def piece_clicked(self, piece):
        if self.heatwaveclock:
        	self.heatwaveclock.cancel()
        self.set_state_vector(piece)
        self.show_heatmap(piece)
    
    def memoize(self,value):
        self.S[0]=value
        return value

    def show_heatmap(self, piece):
        state=np.diag(self.state_vector)
        if piece.piece_type != 'pawn':
        	moveset=piece_moveset[piece.piece_type]
        else:
        	moveset=piece_moveset[f'{piece.piece_color}pawn']
        self.adjacency=0*np.eye(64,dtype="int")
        for rider in moveset[1:]:
        	self.S=[np.eye(64,dtype="int")]
        	collider=state@np.array(rider,dtype="int")
        	if piece.piece_type != 'pawn':
        	    taker=np.diag(self.enemy_territory_vector)@np.array(rider,dtype="int")
        	else:
        		taker=np.diag(self.enemy_territory_vector)@np.array(piece_moveset[f"{piece.piece_color}pawnatk"],dtype="int")
        	if piece.piece_type not in ('pawn','knight','king'):
        		self.adjacency+=(np.eye(64,dtype=int)+taker)@(np.eye(64,dtype=int)+sum([self.memoize(self.S[0]@collider) for i in range(moveset[0])],0*np.eye(64,dtype="int")))-np.eye(64,dtype=int)
        	else:
        		self.adjacency+=(taker+sum([self.memoize(self.S[0]@collider) for i in range(moveset[0])],0*np.eye(64,dtype="int")))
        		
        self.adjacency=np.diag(self.rider_attack_vector)@self.adjacency
        self.adjacency=np.diag(self.single_move_attack_vector)@self.adjacency
        Q=np.diag(np.ones(64,dtype="int")@self.adjacency)
        if self.quantum:
            self.adjacency-=Q
            self.adjacency=self.adjacency@(np.linalg.inv(np.array(Q,dtype="float64")+np.eye(64,dtype="float64")))
        else:
        	self.adjacency=self.adjacency@(np.linalg.inv(np.array(Q,dtype="float64")+np.eye(64,dtype="float64")))
        	
        #self.adjacency=state@np.array(moveset,dtype="int")
        self.S=[np.eye(64,dtype="float64")]
        if self.quantum:
            #self.heat_operator=sum([self.memoize(self.S[0]@self.adjacency*1j/(i+1)) for i in range(100)],np.eye(64,dtype="complex128"))
            self.heat_operator=sci.linalg.expm(self.adjacency*0.2j)
        else:
        	self.heat_operator=sum([self.memoize(self.S[0]@self.adjacency*0.2/(i+1)) for i in range(100)],np.eye(64,dtype="float64"))
        self.heatwaveclock=Clock.schedule_interval(self.heatmapengine,1.0/50.0)
    
    def heatmapengine(self,dt):
        plt.clf()
        self.position_vector=self.heat_operator@self.position_vector
        #heatmap=(self.position_vector).reshape(8,8)
        if self.quantum:
            heatmap=np.array((np.conj(self.position_vector)*self.position_vector),dtype="float64").reshape(8,8)
        else:
        	heatmap=self.position_vector.reshape(8,8)
        #plt.imshow(heatmap/np.argmax(heatmap),cmap='hot',interpolation="nearest")
        plt.imshow(heatmap,cmap='hot',interpolation="nearest")
        #cmap = LinearSegmentedColormap.from_list('custom', ['white', 'red'])
        #plt.imshow(heatmap, cmap=cmap)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = flip(PILImage.open(buf))
        buf.close()
        
        self.canvas.after.clear()
        with self.canvas.after:
            Color(1, 1, 1, 0.8)
            texture = kivy.graphics.texture.Texture.create(size=img.size)
            texture.blit_buffer(img.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            Rectangle(texture=texture, pos=self.pos, size=self.size)
    
    def analysis_toggle(self):
        if self.heatwaveclock:
        	self.heatwaveclock.cancel()
        setattr(self, 'locked', not self.locked)
        #setattr(self.parent.parent.children[2].children[1],'color',(1,1,0,1))
    
    def quantum_toggle(self):
        if self.heatwaveclock:
        	self.heatwaveclock.cancel()
        setattr(self, 'quantum', not self.quantum)

    def clear(self):
        # Clear all pieces, including manually added ones
        if self.heatwaveclock:
        	self.heatwaveclock.cancel()
        setattr(self.parent.parent.children[2].children[1],'state','normal')
        setattr(self,'locked',False)
        pieces_to_remove = [child for child in self.children if isinstance(child, DraggablePiece)]
        
        for piece in pieces_to_remove:
            piece.current_square = None  # Remove reference from the piece's square
            self.remove_widget(piece)  # Remove piece from the board
        
        self.canvas.after.clear()
        self.setup_board()  # Redraw the board background

    def reset_to_start(self):
        # Clear the board first
        self.clear()
        
        # Setup pieces in their starting positions
        piece_setup = {
            # White pieces
            'a1': ('rook', 'white'), 'b1': ('knight', 'white'), 
            'c1': ('bishop', 'white'), 'd1': ('queen', 'white'),
            'e1': ('king', 'white'), 'f1': ('bishop', 'white'),
            'g1': ('knight', 'white'), 'h1': ('rook', 'white'),
            # Black pieces
            'a8': ('rook', 'black'), 'b8': ('knight', 'black'),
            'c8': ('bishop', 'black'), 'd8': ('queen', 'black'),
            'e8': ('king', 'black'), 'f8': ('bishop', 'black'),
            'g8': ('knight', 'black'), 'h8': ('rook', 'black')
        }
        
        # Add pawns
        for file in 'abcdefgh':
            piece_setup[f'{file}2'] = ('pawn', 'white')
            piece_setup[f'{file}7'] = ('pawn', 'black')
            
        # Create and position all pieces
        for pos, (piece_type, color) in piece_setup.items():
            square = self.squares[pos]
            piece = DraggablePiece(piece_type, color, self)
            piece.center = square.center
            piece.current_square = square
            square.occupied_piece = piece
            self.add_widget(piece)

class ChessApp(App):
    def build(self):
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Button row
        button_layout = BoxLayout(size_hint_y=0.1, spacing=10)
        clear_button = Button(text='Clear')
        reset_button = Button(text='New Game')
        lock_button = ToggleButton(text='Analyze Piece')
        quantum_button = ToggleButton(text='Quantum')
        button_layout.add_widget(clear_button)
        button_layout.add_widget(reset_button)
        button_layout.add_widget(lock_button)
        button_layout.add_widget(quantum_button)
        
        # Chess board container
        board_container = BoxLayout(size_hint_y=0.7)
        self.board = ChessBoard(size_hint=(None, None), size=(dp(400), dp(400)))
        board_container.add_widget(self.board)
        
        # Piece selection rows
        piece_rows = BoxLayout(orientation='vertical', size_hint_y=0.2, spacing=5)
        colors = ['white', 'black']
        pieces = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        
        for color in colors:
            row = BoxLayout(spacing=5)
            for piece in pieces:
                piece_widget = DraggablePiece(piece, color, board_ref=self.board, is_template=True)
                row.add_widget(piece_widget)
            piece_rows.add_widget(row)

        # Add all components to main layout
        main_layout.add_widget(button_layout)
        main_layout.add_widget(board_container)
        main_layout.add_widget(piece_rows)

        # Button callbacks
        clear_button.bind(on_press=lambda x: self.board.clear())
        reset_button.bind(on_press=lambda x: self.board.reset_to_start())
        lock_button.bind(on_press=lambda x: self.board.analysis_toggle())
        quantum_button.bind(on_press=lambda x: self.board.quantum_toggle())

        return main_layout

if __name__ == '__main__':
    ChessApp().run()