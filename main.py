from collections import deque
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import os

# Change the current working directory (sometimes it's not where the script is)
os.chdir(os.path.abspath(os.path.dirname(__file__)))


# DO NOT FORGET -- LOWERCASE LETTERS ARE BLACK PIECES, UPPERCASE LETTERS ARE WHITE PIECES

SIZE = 80
FONT = ("Chess Merida Unicode", SIZE//3*2)
COLOURS = (("#6bb3e9", "#b3d4ec"), ("#b58863", "#eed2a6"))[1]
SPRITES = "./pieces.png"


# Chess Board + Game Logics
class Board(tk.Canvas):

    CAPTURED_LIST = {'white':[], 'black':[]}

    FLIP = False  # True - black at bottom, False - White at bottom

    def __init__ (self, parent: 'Main'):
        self.parent = parent
        super().__init__(parent, height=SIZE*8, width=SIZE*8)
        self.board: dict[str,'Piece'] = {}
        self.moving = False
        self.sel_indicator: int|None = None
        self.selected_piece: 'Piece|None' = None
        self.en_passant = None
        self.castling_perms = [True, True, True, True]  # K, Q, k, q
        self.turn = "white"
        self.bind("<Button-1>", self.on_click)
        self.parent.bind("f", lambda _: self.flip_board())
        self.draw_squares()
        self.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    def on_click(self, event): # happens when the user clicks on the canvas
        if self.moving: return
        # find the square that was clicked
        x, y = event.x, event.y
        files, ranks = 'abcdefgh', '87654321'
        file_index = x // SIZE
        rank_index = y // SIZE
        if Board.FLIP:
            file_index = 7 - file_index  # Flip file (a↔h, b↔g, etc.)
            rank_index = 7 - rank_index  # Flip rank (1↔8, 2↔7, etc.)
        if 0 <= file_index <= 8 and 0 <= rank_index < 8:
            square = f"{files[file_index]}{ranks[rank_index]}"
        else: return

        if self.selected_piece: # if a piece is selected
            if self.selected_piece.position == square:
                # if the selected square is the same as the piece's position
                self.deselect()
            elif square in self.board and self.board[square].colour == self.selected_piece.colour:
                # if the selected square has a piece of the same colour
                self.deselect()
                self.selected_piece = self.board[square]
                self.select_visual()
            else:
                self.move_piece(self.selected_piece, square)
        elif square in self.board:
            # if a piece is not selected and the square has a piece
            if self.board[square].colour == self.turn:
                # if the piece belongs to the current player
                self.selected_piece = self.board[square]
                self.select_visual()
            else: return
        else:
            # if a piece is not selected and the square is empty
            self.deselect()

    def move_piece(self, piece: 'Piece', square: str):
        # checks for castling
        if piece.type.lower() == 'k' and piece.position in ('e1', 'e8') and square in ('g1', 'c1', 'g8', 'c8'):
            if square in ('g1', 'g8'):
                self.castling(piece, 'k')
            else:
                self.castling(piece, 'q')
            return

        # piece movement logic
        if not self.move_validation(piece, square):
            self.deselect()
            return
        elif not self.in_check_validation(piece, square):
            self.deselect()
            print(f"King is in check! You cannot move {piece.type} to {square}")
            return
        elif square == self.en_passant:
            self.execute_en_passant(piece)
        if piece.type.lower() == 'p' and (square[1] == '1' or square[1] == '8'):
            # pawn promotion
            target_piece = self.promote(piece)
            original = self.board.pop(piece.position)
            self.board[original.position] = Piece(self, target_piece, original.position)
            piece = self.board[original.position]
            original.remove()

        self.board.pop(piece.position)
        piece.move(square)
        if square in self.board: # if destination square has a piece
            captured = self.board[square]
            Board.CAPTURED_LIST[captured.colour].append(captured)
            captured.remove()
        self.board[square] = piece
        self.deselect()

        # updates castling permissions
        if piece.type == 'K':
            self.castling_perms[0] = False
            self.castling_perms[1] = False
        elif piece.type == 'k':
            self.castling_perms[2] = False
            self.castling_perms[3] = False
        elif piece.type.lower() == 'r':
            if piece.position == 'a1': self.castling_perms[1] = False
            elif piece.position == 'h1': self.castling_perms[0] = False
            elif piece.position == 'a8': self.castling_perms[3] = False
            elif piece.position == 'h8': self.castling_perms[2] = False

        # Check if opponent is in check
        if self.in_check("black" if self.turn == "white" else "white"):
            print(f'{"black" if self.turn == "white" else "white"} is in check!')

        # Change Turn
        self.turn = "black" if self.turn == "white" else "white"

    def select_visual(self):
        # draws a rectangle around the selected piece
        if self.selected_piece is None: return
        x, y = Piece.xycoords(self.selected_piece.position, (2,2))
        self.sel_indicator = self.create_rectangle(x, y, x+SIZE-4, y+SIZE-4,
        fill='', outline='#2f2f2f', width=5)

    def deselect(self):
        # removes the rectangle around the selected piece
        if self.sel_indicator: self.delete(self.sel_indicator)
        self.sel_indicator = None
        self.selected_piece = None

    def draw_squares(self):
        # draws the chess board squares
        for i in range(8):
            for j in range(8):
                self.create_rectangle(SIZE*i, SIZE*j, SIZE*i+SIZE, SIZE*j+SIZE, fill=COLOURS[~(i+j)%2], outline='')

    def load_fen(self, fen: str):
        # loads the board state from a FEN string
        data = fen
        files = 'abcdefgh'
        ranks = data.split('/')
        for rank_index, rank in enumerate(ranks):
            file_index = 0
            for char in rank:
                if char.isdigit():
                    file_index += int(char)
                else:
                    position = f'{files[file_index]}{8-rank_index}'
                    self.board[position] = Piece(self, char, position)
                    file_index += 1

    def execute_en_passant(self, piece: 'Piece'):
        if not self.en_passant: return
        pawn = self.board.pop(f"{self.en_passant[0]}{int(self.en_passant[1])-(-1 if piece.type.islower() else 1)}")
        Board.CAPTURED_LIST[pawn.colour].append(pawn)
        pawn.remove()
        self.en_passant = None

    def castling(self, king: 'Piece', side: str): # side - 'k' or 'q'
        if not self.validate_castling(king, side):
            print("Castling is not allowed")
            self.deselect()
            return

        rank = king.position[1]

        if side == 'k':
            rook_start = f'h{rank}'
            rook_end = f'f{rank}'
            king_target = f'g{rank}'
        else:
            rook_start = f'a{rank}'
            rook_end = f'd{rank}'
            king_target = f'c{rank}'

        # move king
        del self.board[king.position]
        king.move(king_target)
        self.board[king_target] = king

        # move rook
        rook = self.board[rook_start]
        del self.board[rook_start]
        rook.move(rook_end)
        self.board[rook_end] = rook

        # update castling permissions
        if king.type == 'K':
            self.castling_perms[0] = False
            self.castling_perms[1] = False
        else:
            self.castling_perms[2] = False
            self.castling_perms[3] = False

        # change turn + deselect
        self.turn = "black" if self.turn == "white" else "white"
        self.deselect()

    def validate_castling(self, king: 'Piece', side: str):
        # Checks if they are eligible
        index = 0 if king.type.isupper() else 2
        if side == 'q':
            index += 1
        if not self.castling_perms[index]:
            print("King or Rook has moved")
            return False

        # Check if the squares are empty or attacked
        rank = king.position[1]
        if side == 'k':
            if not 'h'+rank in self.board:
                return False
            for file in 'fgh':
                if rank+file in self.board:
                    print("Square is occupied")
                    return False
            for file in 'efg':
                if not self.in_check_validation(king, file+rank):
                    print("King in check or passing through check")
                    return False
        else:
            if not 'a'+rank in self.board:
                return False
            for file in 'abc':
                if rank+file in self.board:
                    print("Square is occupied")
                    return False
            for file in 'cde':
                if not self.in_check_validation(king, file+rank):
                    print("King in check or passing through check")
                    return False

        return True

    def promote(self, pawn: 'Piece'):
        # Displays a dialog to choose the promotion piece
        x, y = Piece.xycoords(pawn.position)
        if (int(pawn.position[1]) == 7 and self.FLIP) or (int(pawn.position[1]) == 2 and not self.FLIP):
            y -= 3*SIZE
        to_return = ctk.StringVar()
        toplevel = ctk.CTkFrame(self.parent)
        #toplevel = ctk.CTkToplevel()
        #toplevel.geometry(f"{SIZE}x{SIZE*4}+{self.winfo_rootx()+x}+{self.winfo_rooty()+y}")
        #toplevel.wm_overrideredirect(True)
        #toplevel.attributes("-topmost", True)
        toplevel.place(x=x, y=y, anchor='nw')
        toplevel.grab_set()
        # ♔♕♖♗♘♙♚♛♜♝♞♟
        button_text = (('Q♕', 'R♖', 'B♗', 'N♘'), ('q♛', 'r♜', 'b♝', 'n♞'))[int(pawn.type.islower())]
        for text in button_text:
            ctk.CTkButton(toplevel, width=SIZE, height=SIZE,
            text=text[1], command=lambda t=text[0]: to_return.set(t),
            font=("FreeSerif", SIZE//3*2), corner_radius=0, fg_color='#F0F0F0',
            hover_color='#606060', text_color='#000000').pack(fill='both')
        self.wait_variable(to_return)
        toplevel.destroy()
        print(to_return.get())
        return to_return.get()

    def move_validation(self, piece: 'Piece', new_coords: str):
        # checks if the new coords are inside bounds
        if not 1 <= int(new_coords[1]) <= 8 or not new_coords[0] in 'abcdefgh':
            return False

        # checks friendly fire
        if new_coords in self.board and piece.colour == self.board[new_coords].colour:
            return False

        # checks if the move is valid
        start, end = piece.position, new_coords
        file_start, rank_start = start[0], int(start[1])
        file_end, rank_end = end[0], int(end[1])

        file_diff = abs(ord(file_end) - ord(file_start))
        rank_diff = abs(rank_end - rank_start)

        set_enpassant = False
        # First - check if the piece is moving according to its rules
        match piece.type.lower():
            case 'p':
                # pawn movement
                # determine the direction of movement
                direction = 1 if piece.type.isupper() else -1
                if file_diff > 1 or rank_diff > 2:
                    #print("file and rank difference")
                    return False # invalid move
                elif int(new_coords[1]) - int(piece.position[1]) not in (direction,2*direction):
                    #print("incorrect direction")
                    return False # pawns can only move forward
                elif rank_diff == 2 and (not (piece.position[1] == '2' or piece.position[1] == '7') or file_diff):
                    #print("move 2 squares - ", rank_diff, piece.position[1])
                    return False # pawns can only move 2 squares from their starting position
                elif (new_coords in self.board and file_diff == 0) or (rank_diff == 2 and f"{file_end}{int(rank_end)-direction}" in self.board):
                    #print("blocking")
                    return False # check if there is a piece in the way
                else:
                    # en passant
                    if (file_diff, rank_diff) == (1,1):
                        if new_coords == self.en_passant:
                            return True
                        elif new_coords not in self.board:
                            return False
                    elif rank_diff == 2:
                        self.en_passant = f"{new_coords[0]}{int(rank_end)-direction}"
                        set_enpassant = True
            case 'k':
                if max(file_diff, rank_diff) > 1:
                    return False # one square in any direction
            case 'q':
                # combination of rook and bishop
                if not (file_diff == 0 or rank_diff == 0 or file_diff == rank_diff):
                    return False # moves in a straight line or diagonally
            case 'r':
                if file_diff and rank_diff:
                    return False # moves ina straight line
            case 'b':
                if not file_diff == rank_diff:
                    return False # moves diagonally
            case 'n':
                if not (file_diff,rank_diff) in ((1,2),(2,1)):
                    return False # L-shaped movement

        # Second - check if the path is clear

        if piece.type.lower() in 'rqb':  # rook, queen or bishop
            """file_step = (end_file - start_file) // max(1, file_diff)  # -1, 0, or 1
            rank_step = (end_rank - start_rank) // max(1, rank_diff)  # -1, 0, or 1
            current_file, current_rank = start_file + file_step, start_rank + rank_step
            while (current_file, current_rank) != (end_file, end_rank):
                square = f"{chr(96 + current_file)}{current_rank}"
                if square in self.board:  # If a piece is found in the path, return False
                    print("blocking piece", square)
                    return False
                current_file += file_step
                current_rank += rank_step"""
            file_step = (ord(file_end) - ord(file_start)) // max(1, file_diff)  # +1, -1, or 0
            rank_step = (rank_end - rank_start) // max(1, rank_diff)  # +1, -1, or 0

            # Start from one step ahead of the starting position
            file, rank = ord(file_start) + file_step, int(rank_start) + rank_step

            # Check all squares along the path (excluding start and end)
            while (file, rank) != (ord(file_end), int(rank_end)):
                square = f"{chr(file)}{rank}"
                if square in self.board:  # A piece is blocking the path
                    return False
                file += file_step
                rank += rank_step

        if not set_enpassant: self.en_passant = None
        return True

    def in_check_validation(self, piece: 'Piece', new_coords: str):
        # Returns whether the move is valid, NOT whether the king is in check
        if piece.position == new_coords: return True

        original_pos = piece.position

        # Simulate the move
        captured_piece = self.board.pop(new_coords, None) # Remove the piece at the destination, if any
        self.board[new_coords] = self.board.pop(original_pos) # Move the piece to the destination
        self.board[new_coords].position = new_coords # Update the piece's position without moving it

        # Check if the king is in check
        in_check = self.in_check(piece.colour)

        # Undo the move
        self.board[original_pos] = self.board.pop(new_coords) # Move the piece back to its original position
        self.board[original_pos].position = original_pos # Update the piece's position without moving it
        if captured_piece: self.board[new_coords] = captured_piece # Restore the captured piece

        return not in_check

    def in_check(self, side: str): # side - 'white' or 'black'
        # check if the king is in check
        for _, piece in self.board.items():
            if piece.type == ('K' if side == 'white' else 'k'):
                king = piece
                break
        for pos, piece in self.board.items():
            if piece.colour == ('white' if side == 'black' else 'black') and self.move_validation(piece, king.position):
                print(f"{piece.type} at {pos} is attacking the king")
                return True
        return False

    def checkmate(self, side: str): # side - 'white' or 'black'
        for _, piece in self.board.items():
            if piece.type.lower() == 'k' and piece.colour == side:
                king = piece
                break
        f, r = king.position
        f = ord(f)
        r = int(r)
        # -1+1 +0+1 +1+1
        # -1+0 +0+0 +1+0
        # -1-1 +0-1 +1-1
        checks = []
        for dx, dy in ((-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)):
            nx, ny = f+dx, r+dy
            new_coords = f"{chr(nx)}{ny}"
            if not self.move_validation(king, new_coords) or not self.in_check_validation(king, new_coords):
                checks.append(False)
            else:
                checks.append(True)
        return any(checks)  # if there is one possible square - then king is safe

    def flip_board(self):
        Board.FLIP = not Board.FLIP
        self.redraw()

    def redraw(self):
        for piece in self.board.values():
            piece.redraw()


# Chess Pieces (reworking to support image based chess pieces)
class Piece:
    # Define the coordinates and size of each piece in the sprite sheet
    PIECE_COORDS = {
        'q': (0, 0), 'k': (1, 0), 'r': (2, 0), 'n': (3, 0), 'b': (4, 0), 'p': (5, 0),
        'Q': (0, 1), 'K': (1, 1), 'R': (2, 1), 'N': (3, 1), 'B': (4, 1), 'P': (5, 1)
    }
    PIECE_SIZE = (60, 60)  # Width and height of each piece in the sprite sheet

    # Load the sprite sheet
    SPRITE_SHEET = Image.open(SPRITES)

    def __init__(self, board: 'Board', type: str, position: str):
        if not type in 'kqbnrpKQBNRP' and len(type) == 1:
            raise TypeError("Invalid Type!")
        self.image = self.get_piece_image(type)
        self.move_list = deque(maxlen=100)
        self.type = type  # black - lowercase, white - uppercase
        self.colour = "black" if type.islower() else "white"
        self.position = position  # Position as chess board cords, e.g. e4
        self.board = board
        self.moving = False
        self.offset = (0, 0)
        self.pid = 0
        self.draw()

    def get_piece_image(self, type: str):
        x, y = Piece.PIECE_COORDS[type]
        piece_image = Piece.SPRITE_SHEET.crop((
            x * Piece.PIECE_SIZE[0],
            y * Piece.PIECE_SIZE[1],
            (x + 1) * Piece.PIECE_SIZE[0],
            (y + 1) * Piece.PIECE_SIZE[1]
        ))
        if SIZE != Piece.PIECE_SIZE[0]:
            piece_image = piece_image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        return ImageTk.PhotoImage(piece_image)

    def __repr__(self) -> str:  # for debugging
        return f"{self.type}@{self.position}"

    def move(self, new_coords: str):
        # piece movement animation
        if self.moving: return
        x0, y0 = self.xycoords(self.position)
        # the offset is used to keep the piece centered on the square
        new_x, new_y = self.xycoords(new_coords, self.offset)
        for interpol_x, interpol_y in self.gen_path(x0, y0, new_x, new_y):
            self.move_list.append((interpol_x, interpol_y))
        self.move_list.append((new_x, new_y))
        self.position = new_coords
        self.update()

    def update(self):
        # moves the piece to the next position in the move list
        if not self.move_list: return
        xto, yto = self.move_list.popleft()
        self.board.moveto(self.pid, xto, yto)
        self.board.after(5, self.update)

    def draw(self):
        # draws the piece on the board
        x1, y1 = self.xycoords(self.position)
        self.pid = self.board.create_image(x1, y1, image=self.image, anchor='nw')
        # get the offset from the bounding box
        bx1, by1, bx2, by2 = self.board.bbox(self.pid)
        gx, gy = self.xycoords(self.position, (SIZE // 2, SIZE // 2))
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        self.offset = (gx - cx, gy - cy)
        self.board.moveto(self.pid, *self.xycoords(self.position, self.offset))

    def redraw(self):
        self.board.delete(self.pid)
        self.draw()

    def remove(self):
        # when the piece is captured
        self.board.delete(self.pid)
        del self

    @staticmethod
    def xycoords(position: str, offset: tuple[int, int] = (0, 0)):
        file = ord(position[0].lower()) - 97
        rank = int(position[1])
        if Board.FLIP:
            file = 7 - file  # Flip file (a↔h, b↔g, etc.)
            rank = 9 - rank  # Flip rank (1↔8, 2↔7, etc.)
        # converts chess board coordinates to tkinter coordinates
        # the offset is used to keep the piece centered on the square
        x = file * SIZE + offset[0]
        y = SIZE * 8 - rank * SIZE + offset[1]
        return x, y

    @staticmethod
    def gen_path(x1, y1, x2, y2, num_points=25):
        # generates a path for the piece movement animation
        for i in range(num_points):
            t = i / (num_points - 1)  # Normalized time (0 to 1)
            ease = 1 - (1 - t) ** 2  # Inverse quadratic easing
            x = x1 + int((x2 - x1) * ease)
            y = y1 + int((y2 - y1) * ease)
            yield x, y


# Info Panel
class Info(ctk.CTkFrame):
    def __init__(self, parent: 'Main'):
        super().__init__(parent, fg_color='transparent')
        self.parent = parent
        self.top()

    def top(self):
        #top_frame = ctk.CTkFrame(self, fg_color='transparent').pack()
        #ctk.CTkButton(self, text='Debug - Print Board', command=self.print_board).pack()
        ctk.CTkButton(self, text='Debug - flip board', command=self.parent.board.flip_board).pack()
        #ctk.CTkButton(self, text='Debug - change turn', command=lambda: setattr(self.parent.board, 'turn', 'black' if self.parent.board.turn == 'white' else 'white')).pack()
        #ctk.CTkButton(self, text='Debug - Get en passant', command=lambda: print(self.parent.board.en_passant)).pack()

    def bottom(self):
        ...
    
    def print_board(self):
        print(self.parent.board.board)


# Main GUI
class Main(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.geometry(f"{SIZE*10}x{SIZE*8}")
        self.title("Tkinter Chess Demo")
        self.board = Board(self)
        info = Info(self)
        self.board.pack(side='left', fill='both')
        info.pack(side='right', fill='both')
        self.mainloop()


Main()

# chess notation
# https://www.chess.com/article/view/chess-notation


## NOTE ##
# When the whole project is complete, do the following tests
# 1. Check if the pieces are moving correctly
# 2. Check if the pieces are capturing correctly
# 3. Make sure castling, en passant, and pawn promotion are working
# 4. Make sure checks and checkmates are detected
# 5. Make sure illegal moves are rejected

# Future improvements
# Add stalemate detection
# Add undo move functionality
# Stockfish integration
# Threefold repetition detection
# 50 moves rule detection

# Depreciated code

"""
@staticmethod
def xycoords(position: str, offset: tuple[int, int] = (0,0)):
    is_flip = Board.FLIP
    # Converts chess board coordinates to tkinter coordinates
    # The offset is used to keep the piece centered on the square
    file = ord(position[0].lower()) - 97
    rank = int(position[1]) - 1

    if is_flip:
        file = 7 - file  # Flip file (a↔h, b↔g, etc.)
        rank = 7 - rank  # Flip rank (1↔8, 2↔7, etc.)

    x = file * SIZE + offset[0]
    y = (7 - rank) * SIZE + offset[1]
    return x, y

def on_click(self, event):  # Happens when the user clicks on the canvas
    if self.moving: 
        return
    
    # Find the square that was clicked
    x, y = event.x, event.y
    files, ranks = 'abcdefgh', '87654321'

    file_index = x // SIZE
    rank_index = y // SIZE

    if Board.FLIP:
        file_index = 7 - file_index  # Flip file selection
        rank_index = 7 - rank_index  # Flip rank selection

    if 0 <= file_index < 8 and 0 <= rank_index < 8:
        square = f"{files[file_index]}{ranks[rank_index]}"
    else: 
        return

     # Rest of logic
"""

# old code for text based chess pieces
"""class Piece:
    # Piece symbols
    PIECES = {name:symbol for name,symbol in zip('KQRBNPkqrbnp',
    ("kqrbnplwtvmo" if FONT[0] == 'Chess Merida' else "♔♕♖♗♘♙♚♛♜♝♞♟" ))}

    def __init__(self, board: Board, type: str, position: str):
        if not type in 'kqbnrpKQBNRP' and len(type) == 1:
            raise TypeError("Invalid Type!")
        self.symbol = Piece.PIECES[type]
        self.move_list = deque(maxlen=100)
        self.type = type # black - lowercase, white - uppercase
        self.colour = "black" if type.islower() else "white"
        self.position = position # Position as chess board cords, e.g. e4
        self.board = board
        self.moving = False
        self.offset = (0,0)
        self.pid = 0
        self.draw()

    def __repr__(self) -> str: # for debugging
        return f"{self.type}@{self.position}"

    def move(self, new_coords: str):
        # piece movement animation
        if self.moving: return
        x0, y0 = self.xycoords(self.position)
        # the offset is used to keep the piece centered on the square
        new_x, new_y = self.xycoords(new_coords, self.offset)
        for interpol_x, interpol_y in self.gen_path(x0, y0, new_x, new_y): 
            self.move_list.append((interpol_x, interpol_y))
        self.move_list.append((new_x, new_y))
        self.position = new_coords
        self.update()

    def update(self):
        # moves the piece to the next position in the move list
        if not self.move_list: return
        xto, yto = self.move_list.popleft()
        self.board.moveto(self.pid, xto, yto)
        self.board.after(5, self.update)

    def draw(self):
        # draws the piece on the board
        x1, y1 = self.xycoords(self.position)
        self.pid = self.board.create_text(x1, y1, font=FONT,
        text=self.symbol, justify='center', anchor='nw')
        # get the offset from the bounding box
        bx1, by1, bx2, by2 = self.board.bbox(self.pid)
        gx, gy = self.xycoords(self.position, (SIZE//2, SIZE//2))
        cx, cy = (bx1+bx2)//2, (by1+by2)//2
        self.offset = (gx-cx, gy-cy)
        self.board.moveto(self.pid, *self.xycoords(self.position, self.offset))

    def redraw(self):
        self.board.delete(self.pid)
        self.draw()

    def remove(self):
        # when the piece is captured
        self.board.delete(self.pid)
        del self

    @staticmethod
    def xycoords(position: str, offset: tuple[int, int] = (0,0)):
        file = ord(position[0].lower()) - 97
        rank = int(position[1])
        if Board.FLIP:
            file = 7 - file  # Flip file (a↔h, b↔g, etc.)
            rank = 9 - rank  # Flip rank (1↔8, 2↔7, etc.)
        # converts chess board coordinates to tkinter coordinates
        # the offset is used to keep the piece centered on the square
        x = file * SIZE + offset[0]
        y = SIZE*8 - rank * SIZE + offset[1]
        return x,y

    @staticmethod
    def gen_path(x1, y1, x2, y2, num_points=25):
        # generates a path for the piece movement animation
        for i in range(num_points):
            t = i / (num_points - 1)  # Normalized time (0 to 1)
            ease = 1 - (1 - t) ** 2  # Inverse quadratic easing
            x = x1 + int((x2 - x1) * ease)
            y = y1 + int((y2 - y1) * ease)
            yield x, y

"""


