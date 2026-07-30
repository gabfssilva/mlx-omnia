/* Recreates Electron's hidden title bar on macOS, and then puts the traffic lights
   where the design wants them. `deno desktop` always reserves a native title-bar
   strip (no API to remove it in 2.9), but the NSWindow lives in this process: set
   fullSizeContentView + transparent title bar straight through AppKit. Direct AppKit
   *mutations* from this thread SIGTRAP, so they go through a KVC dictionary
   dispatched with performSelectorOnMainThread; reads are made in place. */

const FULL_SIZE_CONTENT_VIEW = 1n << 15n
const TITLE_HIDDEN = 1n
const TOOLBAR_UNIFIED = 3n
const SEPARATOR_NONE = 1n

/* Height AppKit gives a unified-toolbar title bar. */
export const TITLEBAR_HEIGHT = 52

/* theme.css zooms the whole page by this factor (see the `html { zoom }` rule there);
   everything below is in native points, so the design's px scale with it. */
const UI_SCALE = 1.25

/* The rail as theme.css draws it: a 76px column whose panel is inset 1px from the
   window's left/top/bottom. The three lights sit inside that panel, centered as a
   group, LIGHTS_TOP below its top edge. */
const RAIL_WIDTH = 76 * UI_SCALE
const PANEL_INSET = 1 * UI_SCALE
const LIGHTS_TOP = 11 * UI_SCALE

export async function hideTitleBar(win: Deno.BrowserWindow): Promise<void> {
  if (Deno.build.os !== 'darwin') return
  try {
    for (let i = 0; i < 50; i++) {
      if (apply()) {
        /* AppKit only expands the content view on the next frame change */
        const [w, h] = win.getSize()
        win.setSize(w, h + 1)
        win.setSize(w, h)
        for (let j = 0; j < 50 && !pinTrafficLights(); j++) {
          await new Promise((r) => setTimeout(r, 100))
        }
        return
      }
      await new Promise((r) => setTimeout(r, 100))
    }
  } catch (e) {
    console.error('hideTitleBar failed (window keeps the native strip):', e)
  }
}

/* With fullSizeContentView the webview covers the whole window and eats the
   title-bar drag. Moving the window ourselves per mousemove flickers (each
   setPosition spawns new mouse events that re-derive the position), so hand the drag
   to AppKit instead: performWindowDragWithEvent: runs the native tracking loop. Our
   Deno callback races the event queue — by the time it fires, NSApp.currentEvent may
   already be a later event — so arm on mousedown and keep trying until a left-button
   event is current.

   What counts as a handle is deliberately small: the drag consumes the click, so it
   may only cover pixels no control claims. Those are the rail's empty top (the
   spacer the lights sit over, whose own clicks AppKit takes before the webview sees
   them) and the thin strip above every view header's padding. */
export function enableWindowDrag(win: Deno.BrowserWindow): void {
  const handle = (x: number, y: number) =>
    x <= RAIL_WIDTH ? y <= 35 * UI_SCALE : y <= 18 * UI_SCALE
  let armed = false
  win.onmousedown = (e) => {
    if (!handle(e.clientX, e.clientY)) return
    armed = !startNativeDrag()
  }
  win.onmousemove = () => {
    if (armed && startNativeDrag()) armed = false
  }
  win.onmouseup = () => {
    armed = false
  }
}

/* AppKit resets the button frames on every layout pass — during a live resize, every
   frame — and repositioning them from here always lands an event-loop tick later, so
   the lights visibly trembled between the two spots. The position is pinned once with
   Auto Layout instead: the buttons leave the autoresizing mask and gain left/top/size
   constraints, which AppKit's own passes then maintain. False while the frames are not
   laid out yet, so the caller can retry; anything else missing leaves the lights where
   AppKit put them rather than throwing. */
export function pinTrafficLights(): boolean {
  if (Deno.build.os !== 'darwin') return true
  try {
    const { S, sel, cls, nsstr, ptrVal } = objc()
    let ready = true
    for (const w of titledWindows()) {
      const buttons = [0n, 1n, 2n].map((kind) =>
        S.msg_id_u64(w, sel('standardWindowButton:'), kind)
      )
      if (buttons.some((button) => button === null)) continue
      /* An --hmr restart finds the constraints already active; a second, identical set
         would only pile up (and conflict once the constants change — those need a full
         restart). */
      if ((S.msg_u64(buttons[0], sel('translatesAutoresizingMaskIntoConstraints')) & 1n) === 0n) {
        continue
      }
      const frames = buttons.map((button) => rect(S.msg_rect(button, sel('frame'))))
      const container = S.msg_id(buttons[0], sel('superview'))
      if (container === null) continue
      const [, , width] = frames[0]
      const pitch = frames[1][0] - frames[0][0]
      if (width <= 0 || pitch <= 0) {
        ready = false
        continue
      }
      const left = PANEL_INSET + (RAIL_WIDTH - PANEL_INSET - (2 * pitch + width)) / 2
      const constraints: Deno.PointerValue[] = []
      for (const [index, button] of buttons.entries()) {
        S.msg_void_ptr_ptr_i8(
          button,
          sel('performSelectorOnMainThread:withObject:waitUntilDone:'),
          sel('setValuesForKeysWithDictionary:'),
          dictionary(S, sel, cls, nsstr, ptrVal, [
            [
              'translatesAutoresizingMaskIntoConstraints',
              S.msg_id_i8(cls('NSNumber'), sel('numberWithBool:'), 0)
            ]
          ]),
          1
        )
        const [, , wide, tall] = frames[index]
        constraints.push(
          S.msg_id_ptr_f64(
            S.msg_id(button, sel('leftAnchor')),
            sel('constraintEqualToAnchor:constant:'),
            S.msg_id(container, sel('leftAnchor')),
            left + index * pitch
          ),
          S.msg_id_ptr_f64(
            S.msg_id(button, sel('topAnchor')),
            sel('constraintEqualToAnchor:constant:'),
            S.msg_id(container, sel('topAnchor')),
            PANEL_INSET + LIGHTS_TOP
          ),
          S.msg_id_f64(S.msg_id(button, sel('widthAnchor')), sel('constraintEqualToConstant:'), wide),
          S.msg_id_f64(S.msg_id(button, sel('heightAnchor')), sel('constraintEqualToConstant:'), tall)
        )
      }
      S.msg_void_ptr_ptr_i8(
        cls('NSLayoutConstraint'),
        sel('performSelectorOnMainThread:withObject:waitUntilDone:'),
        sel('activateConstraints:'),
        S.msg_id_buf_u64(
          cls('NSArray'),
          sel('arrayWithObjects:count:'),
          new BigUint64Array(constraints.map(ptrVal)),
          BigInt(constraints.length)
        ),
        1
      )
    }
    return ready
  } catch (e) {
    console.error('traffic-light pinning failed (AppKit keeps its own):', e)
    return true
  }
}

const LEFT_MOUSE_DOWN = 1n
const LEFT_MOUSE_DRAGGED = 6n

function startNativeDrag(): boolean {
  const { S, sel, cls } = objc()
  const app = S.msg_id(cls('NSApplication'), sel('sharedApplication'))
  const event = S.msg_id(app, sel('currentEvent'))
  if (!event) return false
  const type = S.msg_u64(event, sel('type'))
  if (type !== LEFT_MOUSE_DOWN && type !== LEFT_MOUSE_DRAGGED) return false
  const w = S.msg_id(event, sel('window'))
  if (!w) return false
  S.msg_void_ptr_ptr_i8(
    w,
    sel('performSelectorOnMainThread:withObject:waitUntilDone:'),
    sel('performWindowDragWithEvent:'),
    event,
    0
  )
  return true
}

function rect(bytes: Uint8Array): [number, number, number, number] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  return [
    view.getFloat64(0, true),
    view.getFloat64(8, true),
    view.getFloat64(16, true),
    view.getFloat64(24, true)
  ]
}

function* titledWindows(): Generator<Deno.PointerValue> {
  const { S, sel, cls } = objc()
  const app = S.msg_id(cls('NSApplication'), sel('sharedApplication'))
  const windows = S.msg_id(app, sel('windows'))
  const count = S.msg_u64(windows, sel('count'))
  for (let i = 0n; i < count; i++) {
    const w = S.msg_id_u64(windows, sel('objectAtIndex:'), i)
    if ((S.msg_u64(w, sel('styleMask')) & 1n) !== 0n) yield w
  }
}

type Objc = ReturnType<typeof load>

function dictionary(
  S: Objc['S'],
  sel: Objc['sel'],
  cls: Objc['cls'],
  nsstr: Objc['nsstr'],
  ptrVal: Objc['ptrVal'],
  pairs: [string, Deno.PointerValue][]
): Deno.PointerValue {
  const objects = new BigUint64Array(pairs.map(([, value]) => ptrVal(value)))
  const keys = new BigUint64Array(pairs.map(([key]) => ptrVal(nsstr(key))))
  return S.msg_id_buf_buf_u64(
    cls('NSDictionary'),
    sel('dictionaryWithObjects:forKeys:count:'),
    objects,
    keys,
    BigInt(pairs.length)
  )
}

/* Applies to every titled window still missing the flag; true once at least one titled
   window exists — an --hmr restart finds the flag already set, and the caller must
   still re-run the nudge and the pinning (a no-op once the constraints are in). */
function apply(): boolean {
  const { S, sel, cls, nsstr, ptrVal } = objc()
  let seen = false
  for (const w of titledWindows()) {
    seen = true
    const mask = S.msg_u64(w, sel('styleMask'))
    if ((mask & FULL_SIZE_CONTENT_VIEW) !== 0n) continue
    S.msg_void_ptr_ptr_i8(
      w,
      sel('performSelectorOnMainThread:withObject:waitUntilDone:'),
      sel('setValuesForKeysWithDictionary:'),
      dictionary(S, sel, cls, nsstr, ptrVal, [
        [
          'styleMask',
          S.msg_id_u64(
            cls('NSNumber'),
            sel('numberWithUnsignedLongLong:'),
            mask | FULL_SIZE_CONTENT_VIEW
          )
        ],
        ['titlebarAppearsTransparent', S.msg_id_i8(cls('NSNumber'), sel('numberWithBool:'), 1)],
        ['titleVisibility', S.msg_id_i64(cls('NSNumber'), sel('numberWithLongLong:'), TITLE_HIDDEN)],
        ['toolbarStyle', S.msg_id_i64(cls('NSNumber'), sel('numberWithLongLong:'), TOOLBAR_UNIFIED)],
        [
          'titlebarSeparatorStyle',
          S.msg_id_i64(cls('NSNumber'), sel('numberWithLongLong:'), SEPARATOR_NONE)
        ]
      ]),
      1
    )
    /* An empty unified toolbar makes AppKit lay the title bar out tall
       (TITLEBAR_HEIGHT), which is the space `layoutTrafficLights` positions in. */
    const toolbar = S.msg_id(S.msg_id(cls('NSToolbar'), sel('alloc')), sel('init'))
    S.msg_void_ptr_ptr_i8(
      w,
      sel('performSelectorOnMainThread:withObject:waitUntilDone:'),
      sel('setToolbar:'),
      toolbar,
      1
    )
  }
  return seen
}

let lib: Objc | undefined

function objc() {
  lib ??= load()
  return lib
}

function load() {
  const cstr = (s: string) => new TextEncoder().encode(s + '\0')
  const { symbols: S } = Deno.dlopen('/usr/lib/libobjc.A.dylib', {
    objc_getClass: { parameters: ['buffer'], result: 'pointer' },
    sel_registerName: { parameters: ['buffer'], result: 'pointer' },
    msg_id: { name: 'objc_msgSend', parameters: ['pointer', 'pointer'], result: 'pointer' },
    msg_u64: { name: 'objc_msgSend', parameters: ['pointer', 'pointer'], result: 'u64' },
    msg_id_u64: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'u64'],
      result: 'pointer'
    },
    msg_id_i64: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'i64'],
      result: 'pointer'
    },
    msg_id_i8: { name: 'objc_msgSend', parameters: ['pointer', 'pointer', 'i8'], result: 'pointer' },
    msg_id_buf: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'buffer'],
      result: 'pointer'
    },
    msg_id_buf_buf_u64: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'buffer', 'buffer', 'u64'],
      result: 'pointer'
    },
    /* NSRect comes back by value (four CGFloat). */
    msg_rect: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer'],
      result: { struct: ['f64', 'f64', 'f64', 'f64'] }
    },
    msg_id_f64: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'f64'],
      result: 'pointer'
    },
    msg_id_ptr_f64: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'pointer', 'f64'],
      result: 'pointer'
    },
    msg_id_buf_u64: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'buffer', 'u64'],
      result: 'pointer'
    },
    msg_void_ptr_ptr_i8: {
      name: 'objc_msgSend',
      parameters: ['pointer', 'pointer', 'pointer', 'pointer', 'i8'],
      result: 'void'
    }
  } as const)
  const sel = (s: string) => S.sel_registerName(cstr(s))
  const cls = (s: string) => S.objc_getClass(cstr(s))
  const nsstr = (s: string) => S.msg_id_buf(cls('NSString'), sel('stringWithUTF8String:'), cstr(s))
  const ptrVal = (p: Deno.PointerValue) => BigInt(Deno.UnsafePointer.value(p))
  return { S, sel, cls, nsstr, ptrVal }
}
