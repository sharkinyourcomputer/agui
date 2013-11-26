"Fuckin matplotlib doesn't install easily on windows."
import os

def LinePlotHtml(
    dirpath, filename, title, array_of_data,
    logx=False, xticks_at_A=False):
  logx_str = {False: 'false', True: 'true'}[logx]
  ticks_at_A = [440.0 * 2.0 ** p for p in (-2, -1, 0, 1, 2, 3)]
  hAxis_ticks_str = '' if not xticks_at_A else (
      ', ticks: %s' % ticks_at_A)
  array_of_data_str = '[' + ',\n'.join([
      '[' + ', '.join([repr(x) for x in row]) + ']'
      for row in array_of_data]) + ']'
  with file(os.path.join(dirpath, filename + '.html'), 'w') as f:
    f.write (
'<html>\n<head>\n'
'<script type="text/javascript" src="https://www.google.com/jsapi"></script>\n'
'<script type="text/javascript">\n'
'google.load("visualization", "1", {packages:["corechart"]});\n'
'google.setOnLoadCallback(drawChart);\n'
'function drawChart() {\n'
'var data = google.visualization.arrayToDataTable(\n'
'%(array_of_data_str)s\n'
');\n'
'var options = {\n'
'title: "%(title)s",\n'
'hAxis: {logScale: %(logx_str)s%(hAxis_ticks_str)s}\n'
'};\n'
'var chart = new google.visualization.LineChart(document.getElementById("chart_div"));\n'
'chart.draw(data, options);\n'
'}\n</script>\n</head>\n<body>\n'
'<div id="chart_div" style="width: 900px; height: 500px;"></div>\n'
'</body>\n</html>\n' % locals())


def testLinePlotHtml():
  LinePlotHtml(
      'testout', 'testLinePlotHtml',
      'Number of fucks given.',
      [['time', 'X', 'I'],
       [0, 0, 0],
       [1, 1.38, 0],
       [2, 2.15, 0.95],  # I nearly gave a fuck once.
       [3, 2.03, 0],
       [4, 4.1, 0],
       [5, 2.7, -1.],
       [6, 2.3, -1.5]])

  LinePlotHtml(
      'testout', 'testLinePlotHtmllogxTrue',
      'Number of fucks given.',
      [['time', 'X', 'I'],
       [1e0, 0, 0],
       [1e1, 1.38, 0],
       [1e2, 2.15, 0.95],  # I nearly gave a fuck once.
       [1e3, 2.03, 0],
       [1e4, 4.1, 0],
       [1e5, 2.7, -1.],
       [1e6, 2.3, -1.5]],
      logx=True)



if __name__ == '__main__':
  testLinePlotHtml()
