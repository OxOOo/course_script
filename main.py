# -*- encoding: utf-8 -*-

from mechanize import Browser
import time, cookielib, ssl, re
import httplib, urllib
from bs4 import BeautifulSoup
import sys, os, datetime
sys.path.append(os.curdir)
import captcha as CAP

# ssl._create_default_https_context = ssl._create_unverified_context
reload(sys)
sys.setdefaultencoding('utf-8')

# 读取配置文件
f = open('auth', 'r')
AUTH_USERNAME = f.readline().strip()
AUTH_PASSWORD = f.readline().strip()
f.close()
f = open('courses', 'r')
TERM = f.readline().strip()
BEGIN = f.readline().strip()
line = f.readline().strip()
COURSE_INFOS = []
while len(line) > 0:
    COURSE_INFOS.append(line)
    line = f.readline().strip()
f.close()

def login(browser):
    print 'login'

    LOGIN_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/xklogin.do'
    CAPTCHA_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/login-jcaptcah.jpg?captchaflag=login1'

    browser.open(LOGIN_URL)
    captcha = browser.open(CAPTCHA_URL).read()
    f = open('temp.jpg', 'wb')
    f.write(captcha)
    f.close()

    code = CAP.decode('temp.jpg')
    print 'captcha code:', code
    try:
        data = urllib.urlencode({
            'j_username': AUTH_USERNAME,
            'j_password': AUTH_PASSWORD,
            'captchaflag': 'login1',
            '_login_image_': code
        })
        browser.open("https://zhjwxk.cic.tsinghua.edu.cn/j_acegi_formlogin_xsxk.do", data)
    except httplib.BadStatusLine:
        pass

def search(browser, course_info):
    course_type = course_info.split('-')[0]
    course_number = course_info.split('-')[1]
    course_id = course_info.split('-')[2]
    PAGE_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/xkBks.vxkBksXkbBs.do?m='+course_type+'Search&p_xnxq='+TERM+'&tokenPriFlag='+course_type
    if course_type == 'xwk':
        PAGE_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/xkYjs.vxkYjsXkbBs.do?m='+course_type+'Search&p_xnxq='+TERM+'&tokenPriFlag='+course_type

    if course_type == 'rx': # 任选
        browser.open(PAGE_URL)
        browser.select_form(nr=0)
        browser.form.set_all_readonly(False)
        browser.form['m'] = course_type + 'Search'
        browser.form['page'] = '-1'
        browser.form['is_zyrxk'] = ''
        browser.form['p_kch'] = course_number
        browser.form['p_sort.p1'] = 'bkskyl'
        browser.form['p_sort.asc1'] = 'false'
        browser.form['p_sort.asc2'] = 'true'
        browser.form['p_sort.p2'] = ''
        res = browser.submit()

    elif course_type == 'xx' or course_type == 'bx' or course_type == 'xwk': # 限选，必选
        res = browser.open(PAGE_URL)
    else:
        raise 'unknow course type'

    html = BeautifulSoup(res.read().decode('gbk'), 'html.parser')
    courses = []
    name = 'unknow'
    for row in html.findAll('tr', attrs={'class': "trr2"})+html.findAll('tr', attrs={'class': "trr1"}):
        tds = row.findAll('td')
        if tds[1].text.strip() == course_number and (course_id == 'X' or tds[2].text.strip() == course_id):
            courses.append({
                'course_type': course_type,
                'course_number': tds[1].text.strip(),
                'course_id': tds[2].text.strip(),
                'name': tds[3].text.strip(),
                'remain': tds[4].text.strip(),
                'time': tds[5].text.strip(),
                'teacher': tds[8].text.strip(),
            })
            name = tds[3].text.strip()
    print 'course info:', course_info, 'course number:', course_number, 'course name:', name
    for x in courses: # FIXME
        print x
    return courses

def submit(browser, course):
    course_type = course['course_type']
    PAGE_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/xkBks.vxkBksXkbBs.do?m='+course_type+'Search&p_xnxq='+TERM+'&tokenPriFlag='+course_type
    if course_type == 'xwk':
        PAGE_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/xkYjs.vxkYjsXkbBs.do?m='+course_type+'Search&p_xnxq='+TERM+'&tokenPriFlag='+course_type

    if course_type == 'rx': # 任选
        browser.open(PAGE_URL)
        browser.select_form(nr=0)
        browser.form.set_all_readonly(False)
        browser.form['m'] = course_type+'Search'
        browser.form['page'] = '-1'
        browser.form['is_zyrxk'] = ''
        browser.form['p_kch'] = course['course_number']
        browser.form['p_sort.p1'] = 'bkskyl'
        browser.form['p_sort.asc1'] = 'false'
        browser.form['p_sort.asc2'] = 'true'
        browser.form['p_sort.p2'] = ''
        browser.submit()

    elif course_type == 'xx' or course_type == 'bx' or course_type == 'xwk': # 限选，必选
        browser.open(PAGE_URL)
    else:
        raise 'unknow course type'

    browser.select_form(nr=0)
    browser.form.set_all_readonly(False)
    try:
        if course_type == 'rx':
            browser.form['p_rx_id'] = [TERM + ';' + course['course_number'] + ';' +
                    course['course_id'] + ';']
        else:
            browser.form['p_'+course_type.rstrip('k')+'k_id'] = [TERM + ';' + course['course_number'] + ';' +
                    course['course_id'] + ';']
    except Exception, e:
        print e
        print 'this course has been reserved for freshmen'

    browser.form['m'] = 'save'+course_type[0].upper()+course_type[1]+'Kc'
    print 'submitting <', course['name'], '>'
    res = browser.submit().read().decode('gbk')
    if res.find(u'提交选课成功;') != -1:
        print 'submit success!'
        return True
    else:
        print 'submit fail~'
        d = re.compile(r'showMsg\("(.*)"\)').search(res)
        if d:
            print 'reason:', d.group(1)
        else:
            print 'unable to find reason.'
        return False

total_number = 0
success_number = 0
fail_number = 0
def process(browser, force_not_submit=False):
    global total_number
    global success_number
    global fail_number

    total_number += 1
    print '\n\n============================'
    print 'datetime:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print 'cash:', CAP.getBalance()
    print 'total times:', total_number, 'success times:', success_number, 'fail times:', fail_number

    MAIN_URL = 'http://zhjwxk.cic.tsinghua.edu.cn/xkBks.vxkBksXkbBs.do?m=main'
    while True:
        browser.open(MAIN_URL)
        if browser.geturl() != MAIN_URL:
            login(browser)
        else:
            print 'login success'
            break

    courses = []
    for course_info in COURSE_INFOS:
        courses.extend(search(browser, course_info))
    for course in courses:
        if len(course['remain']) > 0 and int(course['remain']) > 0 and not force_not_submit:
            if submit(browser, course):
                success_number += 1
            else:
                fail_number += 1

def main():
    if len(AUTH_USERNAME) == 0 or len(AUTH_PASSWORD) == 0:
        print 'you need to prepare an auth file, see README.md for more details'
        return
    if len(TERM) == 0 or len(COURSE_INFOS) == 0:
        print 'you need to prepare a courses file, see README.md for more details'
        return
    
    browser = Browser()
    browser.set_handle_redirect(True)
    browser.set_handle_referer(True)
    browser.set_handle_robots(False)
    browser.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
    cj = cookielib.LWPCookieJar()
    browser.set_cookiejar(cj)

    process(browser, True)
    while True:
        try:
            s = datetime.datetime.strptime(BEGIN, '%Y-%m-%d %H:%M')
            n = datetime.datetime.now()
            if n < s:
                print 'remain seconds to start:', (s-n).seconds
                time.sleep((s-n).seconds+1)
                continue
            
            process(browser)
        except Exception, e:
            print e
        time.sleep(2)

if __name__ == '__main__':
    main()
