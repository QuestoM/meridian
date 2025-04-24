import React, { useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Activity, Clock, DollarSign, Users, Tv, BarChart2, Calendar, Clock3 } from 'lucide-react';

// Sample data - in a real implementation, this would come from the backend
const sampleData = {
  breakImpactByPosition: [
    { position: 'Early', retention: 0.92, revenue: 12500 },
    { position: 'Middle', retention: 0.84, revenue: 18300 },
    { position: 'Late', retention: 0.78, revenue: 16700 }
  ],
  breakImpactByDuration: [
    { duration: 'Short (<60s)', retention: 0.91, revenue: 14200 },
    { duration: 'Medium (60-120s)', retention: 0.83, revenue: 19800 },
    { duration: 'Long (>120s)', retention: 0.76, revenue: 23500 }
  ],
  breakImpactByProgramType: [
    { programType: 'News', retention: 0.88, revenue: 15700 },
    { programType: 'Drama', retention: 0.83, revenue: 20100 },
    { programType: 'Comedy', retention: 0.85, revenue: 18400 },
    { programType: 'Documentary', retention: 0.89, revenue: 14900 },
    { programType: 'Reality', retention: 0.76, revenue: 22300 },
    { programType: 'Sports', retention: 0.91, revenue: 25700 }
  ],
  weeklyPlan: {
    optimizedBreaks: [
      { day: 'Monday', totalBreaks: 12, totalRevenue: 58700, avgRetention: 0.85 },
      { day: 'Tuesday', totalBreaks: 11, totalRevenue: 56400, avgRetention: 0.86 },
      { day: 'Wednesday', totalBreaks: 14, totalRevenue: 63200, avgRetention: 0.84 },
      { day: 'Thursday', totalBreaks: 15, totalRevenue: 68500, avgRetention: 0.82 },
      { day: 'Friday', totalBreaks: 18, totalRevenue: 78300, avgRetention: 0.79 },
      { day: 'Saturday', totalBreaks: 10, totalRevenue: 54200, avgRetention: 0.88 },
      { day: 'Sunday', totalBreaks: 9, totalRevenue: 49800, avgRetention: 0.89 }
    ]
  },
  dailyPlan: {
    hourlyBreaks: [
      { hour: '08:00', programType: 'News', retention: 0.90, revenue: 2300, breakDuration: 45 },
      { hour: '09:00', programType: 'Talk Show', retention: 0.87, revenue: 2800, breakDuration: 60 },
      { hour: '10:00', programType: 'Documentary', retention: 0.89, revenue: 2500, breakDuration: 45 },
      { hour: '11:00', programType: 'Reality', retention: 0.82, revenue: 3200, breakDuration: 90 },
      { hour: '12:00', programType: 'News', retention: 0.88, revenue: 3600, breakDuration: 60 },
      { hour: '13:00', programType: 'Talk Show', retention: 0.85, revenue: 3800, breakDuration: 75 },
      { hour: '14:00', programType: 'Drama', retention: 0.83, revenue: 4200, breakDuration: 90 },
      { hour: '15:00', programType: 'Comedy', retention: 0.86, revenue: 4500, breakDuration: 60 },
      { hour: '16:00', programType: 'News', retention: 0.89, revenue: 5100, breakDuration: 60 },
      { hour: '17:00', programType: 'Reality', retention: 0.80, revenue: 5800, breakDuration: 120 },
      { hour: '18:00', programType: 'News', retention: 0.86, revenue: 7200, breakDuration: 75 },
      { hour: '19:00', programType: 'Drama', retention: 0.81, revenue: 8400, breakDuration: 90 },
      { hour: '20:00', programType: 'Reality', retention: 0.78, revenue: 9600, breakDuration: 120 },
      { hour: '21:00', programType: 'Drama', retention: 0.76, revenue: 10800, breakDuration: 90 },
      { hour: '22:00', programType: 'Comedy', retention: 0.79, revenue: 8900, breakDuration: 60 },
      { hour: '23:00', programType: 'News', retention: 0.84, revenue: 6700, breakDuration: 45 }
    ]
  },
  breakDistribution: [
    { name: 'News_Early_Short', value: 12 },
    { name: 'News_Middle_Medium', value: 8 },
    { name: 'Drama_Middle_Medium', value: 15 },
    { name: 'Comedy_Middle_Short', value: 10 },
    { name: 'Reality_Late_Medium', value: 14 },
    { name: 'Sports_Early_Short', value: 6 },
    { name: 'Documentary_Middle_Short', value: 7 },
    { name: 'Other Breaks', value: 17 }
  ]
};

// Colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1'];

const TVBreakDashboard = () => {
  const [activeTab, setActiveTab] = useState('summary');
  const [planningHorizon, setPlanningHorizon] = useState('weekly');
  
  return (
    <div className="bg-gray-50 min-h-screen">
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-6 shadow-md">
        <h1 className="text-3xl font-bold mb-2">TV Commercial Break Optimization Dashboard</h1>
        <p className="text-lg">Make data-driven decisions to balance viewership retention and revenue</p>
      </div>
      
      <div className="p-6">
        {/* Navigation Tabs */}
        <div className="flex mb-6 bg-white rounded-lg shadow-md overflow-hidden">
          <button 
            onClick={() => setActiveTab('summary')} 
            className={`flex items-center px-4 py-3 font-medium ${activeTab === 'summary' ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
          >
            <Activity className="w-5 h-5 mr-2" />
            Impact Analysis
          </button>
          <button 
            onClick={() => setActiveTab('planning')} 
            className={`flex items-center px-4 py-3 font-medium ${activeTab === 'planning' ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
          >
            <Calendar className="w-5 h-5 mr-2" />
            Break Planning
          </button>
          <button 
            onClick={() => setActiveTab('optimization')} 
            className={`flex items-center px-4 py-3 font-medium ${activeTab === 'optimization' ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
          >
            <BarChart2 className="w-5 h-5 mr-2" />
            Optimization
          </button>
        </div>
        
        {/* Main Content */}
        {activeTab === 'summary' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
              <MetricCard 
                title="Average Viewer Retention" 
                value="82.6%" 
                change="+3.2%" 
                icon={<Users className="h-8 w-8 text-blue-500" />} 
              />
              <MetricCard 
                title="Average Revenue per Break" 
                value="₪4,850" 
                change="+8.7%" 
                icon={<DollarSign className="h-8 w-8 text-green-500" />} 
              />
              <MetricCard 
                title="Total Commercial Break Time" 
                value="89 mins/day" 
                change="-5.3%" 
                icon={<Clock className="h-8 w-8 text-orange-500" />} 
              />
              <MetricCard 
                title="Channel Switching Rate" 
                value="14.2%" 
                change="-7.5%" 
                icon={<Tv className="h-8 w-8 text-purple-500" />} 
              />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <ChartCard title="Viewer Retention by Break Position">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sampleData.breakImpactByPosition}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="position" />
                    <YAxis domain={[0.7, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                    <Legend />
                    <Bar dataKey="retention" name="Viewer Retention" fill="#0088FE" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              
              <ChartCard title="Revenue by Break Position">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sampleData.breakImpactByPosition}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="position" />
                    <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} />
                    <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                    <Legend />
                    <Bar dataKey="revenue" name="Revenue" fill="#00C49F" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <ChartCard title="Viewer Retention by Break Duration">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sampleData.breakImpactByDuration}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="duration" />
                    <YAxis domain={[0.7, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                    <Legend />
                    <Bar dataKey="retention" name="Viewer Retention" fill="#FFBB28" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              
              <ChartCard title="Revenue by Break Duration">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sampleData.breakImpactByDuration}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="duration" />
                    <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} />
                    <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                    <Legend />
                    <Bar dataKey="revenue" name="Revenue" fill="#FF8042" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ChartCard title="Viewer Retention by Program Type">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sampleData.breakImpactByProgramType}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="programType" />
                    <YAxis domain={[0.7, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                    <Legend />
                    <Bar dataKey="retention" name="Viewer Retention" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              
              <ChartCard title="Revenue by Program Type">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sampleData.breakImpactByProgramType}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="programType" />
                    <YAxis tickFormatter={(value) => `$${value.toLocaleString()}`} />
                    <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                    <Legend />
                    <Bar dataKey="revenue" name="Revenue" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
          </div>
        )}
        
        {activeTab === 'planning' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-800">Commercial Break Planning</h2>
                <div className="flex space-x-2">
                  <button 
                    onClick={() => setPlanningHorizon('monthly')}
                    className={`px-4 py-2 rounded-md ${planningHorizon === 'monthly' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                  >
                    Monthly
                  </button>
                  <button 
                    onClick={() => setPlanningHorizon('weekly')}
                    className={`px-4 py-2 rounded-md ${planningHorizon === 'weekly' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                  >
                    Weekly
                  </button>
                  <button 
                    onClick={() => setPlanningHorizon('daily')}
                    className={`px-4 py-2 rounded-md ${planningHorizon === 'daily' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                  >
                    Daily
                  </button>
                </div>
              </div>
              
              {planningHorizon === 'weekly' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <MetricCard 
                      title="Total Weekly Breaks" 
                      value="89" 
                      change="-7" 
                      icon={<BarChart2 className="h-8 w-8 text-blue-500" />} 
                    />
                    <MetricCard 
                      title="Total Weekly Revenue" 
                      value="₪429,100" 
                      change="+₪53,200" 
                      icon={<DollarSign className="h-8 w-8 text-green-500" />} 
                    />
                    <MetricCard 
                      title="Avg. Viewer Retention" 
                      value="84.7%" 
                      change="+2.1%" 
                      icon={<Users className="h-8 w-8 text-orange-500" />} 
                    />
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <ChartCard title="Weekly Break Distribution">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={sampleData.weeklyPlan.optimizedBreaks}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="day" />
                          <YAxis yAxisId="left" orientation="left" />
                          <YAxis yAxisId="right" orientation="right" domain={[0.75, 0.9]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                          <Tooltip formatter={(value, name) => {
                            if (name === 'avgRetention') return `${(value * 100).toFixed(1)}%`;
                            return value;
                          }} />
                          <Legend />
                          <Line yAxisId="left" type="monotone" dataKey="totalBreaks" name="Number of Breaks" stroke="#0088FE" />
                          <Line yAxisId="right" type="monotone" dataKey="avgRetention" name="Viewer Retention" stroke="#00C49F" />
                        </LineChart>
                      </ResponsiveContainer>
                    </ChartCard>
                    
                    <ChartCard title="Weekly Revenue by Day">
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={sampleData.weeklyPlan.optimizedBreaks}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="day" />
                          <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
                          <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
                          <Legend />
                          <Bar dataKey="totalRevenue" name="Revenue" fill="#FFBB28" />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartCard>
                  </div>
                </div>
              )}
              
              {planningHorizon === 'daily' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <MetricCard 
                      title="Total Daily Breaks" 
                      value="16" 
                      change="-3" 
                      icon={<BarChart2 className="h-8 w-8 text-blue-500" />} 
                    />
                    <MetricCard 
                      title="Total Daily Revenue" 
                      value="₪87,400" 
                      change="+₪9,800" 
                      icon={<DollarSign className="h-8 w-8 text-green-500" />} 
                    />
                    <MetricCard 
                      title="Avg. Break Duration" 
                      value="75 sec" 
                      change="-15 sec" 
                      icon={<Clock3 className="h-8 w-8 text-orange-500" />} 
                    />
                  </div>
                  
                  <ChartCard title="Hourly Break Distribution">
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={sampleData.dailyPlan.hourlyBreaks}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="hour" />
                        <YAxis yAxisId="left" orientation="left" tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
                        <YAxis yAxisId="right" orientation="right" domain={[0.7, 0.95]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                        <Tooltip formatter={(value, name) => {
                          if (name === 'retention') return `${(value * 100).toFixed(1)}%`;
                          if (name === 'revenue') return `$${value.toLocaleString()}`;
                          return value;
                        }} />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="revenue" name="Revenue" stroke="#0088FE" />
                        <Line yAxisId="right" type="monotone" dataKey="retention" name="Viewer Retention" stroke="#00C49F" />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartCard>
                  
                  <ChartCard title="Break Durations by Hour">
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={sampleData.dailyPlan.hourlyBreaks}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="hour" />
                        <YAxis tickFormatter={(value) => `${value}s`} />
                        <Tooltip formatter={(value) => `${value} seconds`} />
                        <Legend />
                        <Bar dataKey="breakDuration" name="Break Duration" fill="#FFBB28" />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartCard>
                </div>
              )}
              
              {planningHorizon === 'monthly' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <MetricCard 
                      title="Total Monthly Breaks" 
                      value="356" 
                      change="-28" 
                      icon={<BarChart2 className="h-8 w-8 text-blue-500" />} 
                    />
                    <MetricCard 
                      title="Total Monthly Revenue" 
                      value="₪1,758,400" 
                      change="+₪215,600" 
                      icon={<DollarSign className="h-8 w-8 text-green-500" />} 
                    />
                    <MetricCard 
                      title="Avg. Daily Breaks" 
                      value="11.9" 
                      change="-0.9" 
                      icon={<Calendar className="h-8 w-8 text-orange-500" />} 
                    />
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <ChartCard title="Break Type Distribution">
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={sampleData.breakDistribution}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            {sampleData.breakDistribution.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip formatter={(value) => value} />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </ChartCard>
                    
                    <ChartCard title="Program Type Impact">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={sampleData.breakImpactByProgramType}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="programType" />
                          <YAxis yAxisId="left" orientation="left" tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} />
                          <YAxis yAxisId="right" orientation="right" domain={[0.7, 0.95]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                          <Tooltip formatter={(value, name) => {
                            if (name === 'retention') return `${(value * 100).toFixed(1)}%`;
                            if (name === 'revenue') return `$${value.toLocaleString()}`;
                            return value;
                          }} />
                          <Legend />
                          <Line yAxisId="left" type="monotone" dataKey="revenue" name="Revenue" stroke="#0088FE" />
                          <Line yAxisId="right" type="monotone" dataKey="retention" name="Viewer Retention" stroke="#00C49F" />
                        </LineChart>
                      </ResponsiveContainer>
                    </ChartCard>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'optimization' && (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-md p-6 mb-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Optimization Recommendations</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <RecommendationCard 
                  title="Reduce Break Duration"
                  program="Reality Shows"
                  impact="+5.2% Retention"
                  recommendation="Reduce from 120s to 90s"
                  priority="High"
                />
                <RecommendationCard 
                  title="Reposition Breaks"
                  program="Drama Series"
                  impact="+₪3,200 Revenue"
                  recommendation="Move from Late to Middle position"
                  priority="Medium"
                />
                <RecommendationCard 
                  title="Consolidate Breaks"
                  program="News Programs"
                  impact="+4.1% Retention"
                  recommendation="Fewer but slightly longer breaks"
                  priority="High"
                />
                <RecommendationCard 
                  title="Optimize Prime Time"
                  program="All Prime Time"
                  impact="+₪11,500 Revenue"
                  recommendation="Standard 90s breaks at plot transitions"
                  priority="High"
                />
              </div>
              
              <div className="space-y-6">
                <h3 className="text-lg font-bold text-gray-700 mb-3">Break Optimization Scenarios</h3>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white">
                    <thead>
                      <tr className="bg-gray-100 border-b">
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Scenario</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Revenue Impact</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Retention Impact</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Break Changes</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recommendation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Revenue Maximization</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">+₪34,700</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600">-3.2%</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">+5 breaks, longer durations</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Short-term only</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Retention Maximization</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600">-₪22,300</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">+7.1%</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">-8 breaks, shorter durations</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Special programming</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Balanced Optimization</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">+₪15,800</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">+2.4%</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Repositioned breaks, optimized durations</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">Recommended</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Weekend Focus</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">+₪9,400</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600">+1.8%</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Optimized weekend scheduling</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Consider for sports events</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const MetricCard = ({ title, value, change, icon }) => {
  const isPositive = change.startsWith('+');
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-medium text-gray-700">{title}</h3>
        {icon}
      </div>
      <p className="text-3xl font-bold text-gray-900 mb-2">{value}</p>
      <p className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
        {change} vs. previous period
      </p>
    </div>
  );
};

const ChartCard = ({ title, children }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>
      {children}
    </div>
  );
};

const RecommendationCard = ({ title, program, impact, recommendation, priority }) => {
  let priorityColor = 'bg-yellow-100 text-yellow-800';
  if (priority === 'High') priorityColor = 'bg-red-100 text-red-800';
  if (priority === 'Low') priorityColor = 'bg-green-100 text-green-800';
  
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
      <div className="flex justify-between items-start mb-2">
        <h3 className="text-md font-medium text-gray-900">{title}</h3>
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${priorityColor}`}>{priority}</span>
      </div>
      <p className="text-sm text-gray-600 mb-1">Program: <span className="font-medium">{program}</span></p>
      <p className="text-sm text-green-600 font-medium mb-3">{impact}</p>
      <div className="mt-2 pt-2 border-t border-gray-200">
        <p className="text-sm text-gray-700">{recommendation}</p>
      </div>
    </div>
  );
};

export default TVBreakDashboard;
