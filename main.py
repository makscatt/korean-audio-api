import asyncio
import os
import json
import logging
import sys
import time
import io
from PIL import Image, ImageDraw, ImageOps
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile, InputMediaPhoto, BufferedInputFile

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

API_TOKEN = os.environ.get("TELEGRAM_API_TOKEN")

COORDS = [
    (512, 184), 
    (400, 430), 
    (600, 430), 
    (300, 750), 
    (700, 750), 
    (230, 1070),
    (800, 1070) 
]

IMG_SIZES = [
    (400, 400), 
    (450, 450), 
    (450, 450), 
    (500, 500), 
    (500, 500), 
    (550, 550), 
    (550, 550)  
]

with open("candidates.json", "r", encoding="utf-8") as f:
    CANDIDATES_DATA = json.load(f)

CANDIDATES_LIST = [{"id": k, "name": v} for k, v in CANDIDATES_DATA.items()]

bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

class SelectionStates(StatesGroup):
    selecting = State()

def create_tree_image(selected_ids):
    base = Image.open("img/tree.png").convert("RGBA")
    
    try:
        santa_img = Image.open("img/final_santa.png").convert("RGBA")
        santa_size = (700, 700)
        santa_coords = (512, 1251)
        
        santa_img = ImageOps.fit(santa_img, santa_size, centering=(0.5, 0.5))
        x_santa = santa_coords[0] - santa_size[0] // 2
        y_santa = santa_coords[1] - santa_size[1] // 2
        
        base.paste(santa_img, (x_santa, y_santa), santa_img)
    except Exception as e:
        logging.error(f"Error processing final_santa: {e}")

    for i, uid in enumerate(selected_ids):
        if i >= len(COORDS): break
        
        try:
            current_size = IMG_SIZES[i]
            actor_img = Image.open(f"img/{uid}.png").convert("RGBA")
            actor_img = ImageOps.fit(actor_img, current_size, centering=(0.5, 0.5))
            
            x = COORDS[i][0] - current_size[0] // 2
            y = COORDS[i][1] - current_size[1] // 2
            
            base.paste(actor_img, (x, y), actor_img)
        except Exception as e:
            logging.error(f"Error processing image {uid}: {e}")

    bio = io.BytesIO()
    base.save(bio, format="PNG")
    bio.seek(0)
    return bio

async def reminder_timer(user_id: int, state: FSMContext, timestamp: float):
    await asyncio.sleep(900)
    current_state = await state.get_state()
    data = await state.get_data()
    
    if current_state == SelectionStates.selecting.state:
        last_time = data.get("last_action_time", 0)
        if last_time == timestamp:
            await bot.send_message(user_id, "Эй! Ты ещё тут? Осталось выбрать совсем немного!")

async def build_keyboard(all_items, selected_ids, page):
    items_per_page = 15
    available_items = [item for item in all_items if item["id"] not in selected_ids]
    
    start_index = page * items_per_page
    end_index = start_index + items_per_page
    page_items = available_items[start_index:end_index]
    
    builder = InlineKeyboardBuilder()
    
    for item in page_items:
        builder.button(text=item["name"], callback_data=f"pick:{item['id']}")
    
    builder.adjust(3)

    nav_buttons = []
    if page > 0:
        nav_buttons.append(types.InlineKeyboardButton(text="⬅️", callback_data=f"page:{page-1}"))
    if end_index < len(available_items):
        nav_buttons.append(types.InlineKeyboardButton(text="➡️", callback_data=f"page:{page+1}"))
    
    if nav_buttons:
        builder.row(*nav_buttons)
        
    return builder.as_markup()

@dp.message(Command("start"))
@dp.message(F.text.lower() == "привет")
async def start(message: types.Message, state: FSMContext):
    await state.clear()
    await state.set_state(SelectionStates.selecting)
    
    timestamp = time.time()
    await state.update_data(selected=[], page=0, last_action_time=timestamp)
    
    asyncio.create_task(reminder_timer(message.from_user.id, state, timestamp))
    
    keyboard = await build_keyboard(CANDIDATES_LIST, [], 0)
    
    photo = FSInputFile("img/santa.png")
    
    text = (
        "Кто из корейских красавчиков станет идеальным украшением для твоей новогодней елочки?\n\n"
        "Это будет трудный выбор. Всего в списке <b>45 красавчиков</b>, а выбрать нужно только 7.\n\n"
        "👇 <i>Используй кнопки внизу (⬅️ ➡️), чтобы листать страницы и посмотреть всех!</i>"
    )
    
    await message.answer_photo(photo=photo, caption=text, reply_markup=keyboard)

@dp.callback_query(lambda c: c.data.startswith("page:"))
async def process_page(callback: types.CallbackQuery, state: FSMContext):
    await callback.answer()
    
    page = int(callback.data.split(":")[1])
    data = await state.get_data()
    selected = data.get("selected", [])
    
    timestamp = time.time()
    await state.update_data(page=page, last_action_time=timestamp)
    asyncio.create_task(reminder_timer(callback.from_user.id, state, timestamp))

    keyboard = await build_keyboard(CANDIDATES_LIST, selected, page)
    
    try:
        await callback.message.edit_reply_markup(reply_markup=keyboard)
    except:
        pass

@dp.callback_query(lambda c: c.data.startswith("pick:"))
async def process_pick(callback: types.CallbackQuery, state: FSMContext):
    item_id = callback.data.split(":")[1]
    data = await state.get_data()
    selected = data.get("selected", [])
    page = data.get("page", 0)

    if item_id not in selected:
        selected.append(item_id)
    
    timestamp = time.time()
    await state.update_data(selected=selected, last_action_time=timestamp)
    asyncio.create_task(reminder_timer(callback.from_user.id, state, timestamp))

    selected_names = [CANDIDATES_DATA[sid] for sid in selected]

    if len(selected) >= 7:
        processing_media = InputMediaPhoto(
            media=FSInputFile("img/processing.png"),
            caption="🎄 Наряжаем елочку... Подожди немного!"
        )
        await callback.message.edit_media(media=processing_media, reply_markup=None)

        await asyncio.sleep(3)

        loop = asyncio.get_running_loop()
        result_img_io = await loop.run_in_executor(None, create_tree_image, selected)
        result_file = BufferedInputFile(result_img_io.read(), filename="result.png")
        
        names_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(selected_names)])

        result_text = (
            "<b>Твоя елочка украшена! 🎄🎅🏻</b>\n\n"
            "В этом году тебя будут радовать:\n\n"
            f"{names_list}\n\n"
            "Поделись Ёлочкой у меня в группе: https://t.me/KoreanMaks"
        )

        new_media = InputMediaPhoto(
            media=result_file,
            caption=result_text
        )
        
        await callback.message.edit_media(media=new_media, reply_markup=None)
        await state.clear()
    else:
        remaining = 7 - len(selected)
        current_text = ", ".join(selected_names)
        
        available_count_now = len([item for item in CANDIDATES_LIST if item["id"] not in selected])
        max_pages = (available_count_now - 1) // 15
        if page > max_pages:
            page = max_pages

        await state.update_data(page=page)
        
        keyboard = await build_keyboard(CANDIDATES_LIST, selected, page)
        
        text = (
            "Кто из корейских красавчиков станет идеальным украшением для твоей новогодней елочки?\n\n"
            f"Выбрано: <b>{current_text}</b>\n"
            f"Осталось выбрать: <b>{remaining}</b>\n\n"
            "👇 <i>Листай страницы (⬅️ ➡️), чтобы найти всех кандидатов!</i>"
        )
        
        await callback.message.edit_caption(caption=text, reply_markup=keyboard)

@dp.message()
async def handle_any_text(message: types.Message):
    await message.answer("Чтобы украсить Ёлку нажми /start \n\n Или напиши 'привет'")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())