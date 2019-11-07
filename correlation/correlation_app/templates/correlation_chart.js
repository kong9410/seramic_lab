$(document).ready(function () {
    $.ajax({
        url: 'api/corr_data',
        datatype: 'json',
        success: function (data) {
            console.log(data)
            make_chart(data);
        },
        error: function () {
            console.error("failed to load data");
        }
    })
})

function make_chart(data) {
    //console.log(line)
    var xs = {};
    xs[data['y_label'] + '/' + data['x_label']] = 'x_data';
    xs['Regression'] = 'x_fit';
    console.log(xs);
    var chart = c3.generate({
        bindto: "#scatterplot",
        data: {
            xs: xs,
            // iris data from R
            columns: [
                data['x_data'],
                data['y_data'],
                data['x_fit'],
                data['y_quad']
            ],
            type: 'scatter',
            types: {
                'x_fit': 'line',
                'Regression': 'line'
            }
        },
        axis: {
            x: {
                label: data['x_label'],
                tick: {
                    fit: false
                }
            },
            y: {
                label: data['y_label']
            }
        }
    });
}